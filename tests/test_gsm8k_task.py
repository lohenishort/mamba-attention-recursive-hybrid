import pytest
from pathlib import Path

from mamba_hybrid.config import MambaHybridConfig
from mamba_hybrid.tasks.gsm8k import (
    BOS,
    EOS,
    decode_bytes,
    encode_answer,
    encode_bytes,
    extract_answer,
    normalize_answer,
)
from scripts.train_gsm8k import GSM8KDataset, GSM8KReasoningModel, collate_gsm8k


def test_gsm8k_extracts_only_normalized_final_answer() -> None:
    assert extract_answer("Work shown here. #### 1,024") == "1024"
    assert normalize_answer("-0007") == "-7"


def test_gsm8k_rejects_malformed_answers() -> None:
    with pytest.raises(ValueError, match="####"):
        extract_answer("42")
    with pytest.raises(ValueError, match="not an integer"):
        normalize_answer("3.5")


def test_gsm8k_byte_tokenizer_round_trips_utf8() -> None:
    encoded = encode_bytes("café")

    assert encoded[-1] == EOS
    assert decode_bytes(encoded) == "café"


def test_gsm8k_answer_encoding_is_shifted_for_teacher_forcing() -> None:
    decoder_input, target = encode_answer("-12")

    assert decoder_input[0] == BOS
    assert target[-1] == EOS
    assert decode_bytes(target) == "-12"


def test_gsm8k_dataset_and_model_use_dynamic_masked_batches(tmp_path: Path) -> None:
    dataset_path = tmp_path / "gsm8k.jsonl"
    dataset_path.write_text(
        '{"question":"1+1?","answer":"Add. #### 2"}\n'
        '{"question":"10-3?","answer":"Subtract. #### 7"}\n'
    )
    dataset = GSM8KDataset(str(dataset_path), max_question_bytes=32)
    questions, question_mask, decoder_inputs, targets = collate_gsm8k(
        [dataset[0], dataset[1]]
    )
    config = MambaHybridConfig(
        d_model=8,
        n_meta=2,
        l_ans=1,
        n_steps=1,
        M_min=1,
        M_max=1,
        vocab_size=259,
    )
    model = GSM8KReasoningModel(config, max_question_bytes=32, max_answer_length=8)

    logits, probabilities = model(questions, question_mask, decoder_inputs)

    assert logits.shape == (*targets.shape, 259)
    assert len(probabilities) == 1
    assert question_mask.sum(dim=1).tolist() == [5, 6]
