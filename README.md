# Piper TTS for NVDA

High-quality neural text-to-speech addon for NVDA using Piper TTS engine.

## Features

- **High-Quality Speech**: Neural text-to-speech with natural-sounding voice
- **Ultra-Low Latency**: Optimized for <50ms response time
- **Character Speech Support**: Speaks individual typed characters
- **Rate Control**: Adjustable speech rate in NVDA settings
- **Intelligent Caching**: Pre-caches common words for instant playback
- **Daemon Pool Architecture**: Multiple background processes for parallel synthesis

## Installation

1. Copy the addon folder to NVDA's addons directory
2. Restart NVDA
3. Go to NVDA Settings > Speech > Synthesizer and select "Piper TTS"

## Voice Model

This addon uses the `en_us-jane-medium.onnx` voice model for high-quality English speech synthesis.

## Performance

- **Synthesis Speed**: <50ms for typical phrases
- **Cache Hit Rate**: Instant playback for common words
- **Audio Quality**: 22050 Hz, 16-bit mono PCM
- **Real-time Factor**: Typically 0.1-0.3x (synthesis faster than playback)

## ⚠️ IMPORTANT LIMITATION

**Do not close Command Prompt (cmd.exe) windows while using this addon, as it will crash NVDA.**

This is a Windows system limitation where the piper.exe processes are linked to console sessions. We have attempted extensive process isolation but this appears to be unfixable at the application level.

**Recommended workaround**: Use PowerShell instead of cmd, or avoid using command-line tools while NVDA is active with Piper TTS.

## Technical Details

- **Engine**: Piper TTS with ONNX neural network inference
- **Architecture**: Multi-daemon pool with intelligent load balancing
- **Audio**: Windows Audio Session API (WASAPI) via nvwave
- **Caching**: LRU cache with rate-aware keys
- **Threading**: Lock-free synthesis patterns for ultra-low latency

## Troubleshooting

- Check NVDA logs for detailed synthesis timing information
- If speech becomes slow, restart NVDA to reset the daemon pool

## Support

This is a research/development addon. For issues, check the NVDA logs and `KNOWN_ISSUES.md` file.
