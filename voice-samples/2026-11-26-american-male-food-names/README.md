# American male voice listening test

Exact text for all three options: “Chicken breast. Greek yogurt. Peanut butter.”

- Michael: Kokoro am_michael
- Liam: Kokoro am_liam
- Eric: Kokoro am_eric

The prior muscle Reel used Kokoro v1.0 af_heart (female), fixed 103-token voice conditioning. These male tests instead use each voice’s native conditioning for the complete text, synthesized as one inference batch at speed 1.0. The whole three-item recording supplies context; split clips demonstrate extraction from that recording instead of isolated-word synthesis. No added spoken words. Full comparison files are the listening standard; split clips are provisional until choice and video timing review.

Processing: constant gain toward -18 LUFS, capped to avoid peaks above -2dBTP; 6ms recording-edge fades and 4ms split-edge fades. No denoise, compression, limiter, time stretch, or pitch shift. Native vocal dynamics preserved. MP3 192kbps plus 24-bit PCM WAV at 24kHz. Automated transcription verifies all six words in each combined sample. Subjective naturalness and perceived age need listener judgment; these presets do not support reliable age or expressive-style instructions.

No paid APIs called. No OpenAI or ElevenLabs API credential is configured; plugin discovery did not expose a direct connected speech-generation tool for either provider. The installed fal integration is pay-as-you-go, and no generation was submitted. Accordingly this is a comparison of actual free Kokoro outputs, not an audio comparison with unavailable paid providers.

Model: Kokoro v1.0 82M, Apache-2.0: https://huggingface.co/hexgrad/Kokoro-82M
Runner: kokoro-onnx 0.6.1, MIT: https://github.com/thewh1teagle/kokoro-onnx

No full video rendered or narration replaced. Approved title y140 retained in working composition. After selection: item names only (except the separately requested opening question); original presenter ratings and explanations; native moving footage during announcements, no still frames.
