from pathlib import Path


class Qwen3ASR:
    parallel_safe = False  # one GPU model instance; batching happens inside

    def __init__(self, repo: str, language: str = "en"):
        # PIN AT EXECUTION (verified 2026-07-05 against the model cards):
        # https://huggingface.co/Qwen/Qwen3-ASR-1.7B is served through the
        # separate `qwen-asr` PyPI package (Qwen3ASRModel), not transformers.
        # Native transformers support lives on the "-hf" checkpoints, e.g.
        # https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf, added in
        # huggingface/transformers#43838 and first shipped in transformers
        # v5.13.0 (released 2026-07-03; no prior tag contains it). That card
        # recommends AutoProcessor + AutoModelForMultimodalLM driven through
        # processor.apply_transcription_request(...), not
        # pipeline("automatic-speech-recognition", ...). We normalize the
        # repo id to the "-hf" variant so this class still accepts the bare
        # "Qwen/Qwen3-ASR-*" ids used by the registry.
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.name = repo.rsplit("/", 1)[-1].lower()
        self.language = language
        hf_repo = repo if repo.endswith("-hf") else f"{repo}-hf"
        self._processor = AutoProcessor.from_pretrained(hf_repo)
        self._model = AutoModelForMultimodalLM.from_pretrained(
            hf_repo, dtype=torch.bfloat16, device_map="auto"
        )

    def transcribe(self, audio_path: Path) -> str:
        inputs = self._processor.apply_transcription_request(
            audio=str(audio_path), language=self.language
        ).to(self._model.device, self._model.dtype)
        output_ids = self._model.generate(**inputs, max_new_tokens=256)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        return self._processor.decode(generated_ids, return_format="transcription_only")[0]
