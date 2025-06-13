# -*- coding: UTF-8 -*-
# A part of the Piper TTS addon for NVDA
# Copyright (C) 2024 Piper NVDA Contributors
# This file is covered by the GNU General Public License.

"""
Piper TTS synthDriver for NVDA
High-quality neural text-to-speech using Piper TTS engine
"""

import os
import sys
import threading
import time
import tempfile
import subprocess
import json
import struct
from typing import Optional, Dict, Any

import nvwave
from synthDriverHandler import SynthDriver, VoiceInfo, synthIndexReached, synthDoneSpeaking
from logHandler import log
import addonHandler

# Fix queue import for Python 2/3 compatibility
try:
    import queue
except ImportError:
    import Queue as queue

from speech.types import SpeechSequence
from speech.commands import (
    IndexCommand,
    CharacterModeCommand,
    LangChangeCommand,
    BreakCommand,
    PitchCommand,
    RateCommand,
    VolumeCommand,
    PhonemeCommand,
)

# Get addon directory
addon = addonHandler.getCodeAddon()
ADDON_DIR = addon.path if addon else None


class SynthDriver(SynthDriver):
    name = "piperTTS"
    description = "Piper TTS - Neural Text-to-Speech"

    supportedSettings = (
        SynthDriver.RateSetting(),
        SynthDriver.VoiceSetting(),
        SynthDriver.VolumeSetting(),
    )
    
    supportedCommands = {
        IndexCommand,
        CharacterModeCommand,
        BreakCommand,
        RateCommand,
        VolumeCommand,
        PitchCommand,
        PhonemeCommand,
    }
    
    supportedNotifications = {synthIndexReached, synthDoneSpeaking}
    
    # Character speech support
    @property
    def supportsCharacterSpelling(self):
        """Support for character-by-character speech"""
        return True
    
    @property
    def supportedCharacterModes(self):
        """Support different character modes"""
        return ["off", "talk"]
    
    # lastIndex is inherited from base SynthDriver class - don't override

    # Voice model configuration - hardcoded to en_us-jane-medium
    _VOICE_MODEL = "en_us-jane-medium"
    _MODEL_FILE = "en_us-jane-medium.onnx"
    _CONFIG_FILE = "en_us-jane-medium.onnx.json"

    def __init__(self):
        super().__init__()
        self._speaking = False
        self._current_index = 0
        self._rate = 50  # Default rate (0-100 scale)
        self._volume = 75  # Default volume (0-100 scale)
        
        # Available voices - hardcoded to Jane
        self._voices = {
            "jane": VoiceInfo("jane", "Jane (Medium Quality)", "en-US")
        }
        self._current_voice = "jane"
        
        # Ultra-low-latency mode - minimal locking
        self._piper_voice = None
        self._player = None
        self._piper_process = None  # For daemon mode
        self._daemon_lock = threading.RLock()  # Use RLock for better performance
        self._player_lock = threading.RLock()  # Use RLock for better performance
        
        # Single shared player to prevent overlapping voices
        self._shared_player = None
        self._shared_player_lock = threading.Lock()
        
        # Ultra-fast synthesis system for <10ms response
        self._synthesis_queue = queue.Queue()
        self._synthesis_thread = None
        self._synthesis_thread_running = False
        
        # Pre-synthesis cache for instant responses
        self._audio_cache = {}
        self._cache_lock = threading.Lock()
        self._max_cache_size = 100
        
        # Multiple daemon processes for parallel synthesis
        self._daemon_pool = []
        self._daemon_pool_lock = threading.Lock()
        self._max_daemons = 2  # Reduce to 2 for stability
        
        # Pre-allocated response buffers
        self._response_buffers = []
        self._buffer_lock = threading.Lock()
        
        # Daemon cleanup timer - keep daemon alive longer for better performance
        self._daemon_cleanup_timer = None
        self._daemon_idle_timeout = 120.0  # Stop daemon after 2 minutes of inactivity
        
        # Start synthesis worker thread
        self._start_synthesis_worker()
        
        # Start daemon pool for ultra-low latency (replaces single daemon)
        self._start_daemon_pool()
        
        # Start daemon health monitor with crash recovery
        self._start_daemon_monitor()
        
        # Start audio session crash recovery monitor
        self._start_audio_session_monitor()
        
        # Pre-cache common words for instant synthesis
        self._pre_cache_common_words()
        
        # Don't start the single daemon since we're using a pool
        self._use_daemon_pool = True
        
        log.info("Piper TTS driver initialized successfully - Ultra-low latency neural TTS ready")
        

    def _start_synthesis_worker(self):
        """Start the synthesis worker thread for sequential playback"""
        if not self._synthesis_thread_running:
            self._synthesis_thread_running = True
            self._synthesis_thread = threading.Thread(target=self._synthesis_worker, daemon=True)
            self._synthesis_thread.start()

    def _synthesis_worker(self):
        """Worker thread that processes synthesis requests sequentially"""
        while self._synthesis_thread_running:
            try:
                # Get next synthesis task with timeout
                task = self._synthesis_queue.get(timeout=1.0)
                if task is None:  # Shutdown signal
                    break
                    
                text, index = task
                
                # Check if we're still supposed to be speaking
                if not self._speaking:
                    continue
                
                # Get or create shared player
                with self._shared_player_lock:
                    if not self._shared_player:
                        try:
                            self._shared_player = nvwave.WavePlayer(
                                channels=1,
                                samplesPerSec=22050,
                                bitsPerSample=16
                            )
                        except Exception as e:
                            log.error(f"Failed to create shared player: {e}")
                            continue
                
                # Ultra-fast synthesis with multiple approaches
                audio_data = None
                start_time = time.time()
                self._last_synthesis_time = start_time  # Track for crash detection
                
                log.debug(f"Starting synthesis for: {text[:30]}...")
                
                # 1. Try cached audio first (instant response)
                cached_audio = self._get_cached_audio(text)
                if cached_audio:
                    log.info(f"INSTANT cache hit for: {text[:30]}... (0ms)")
                    audio_data = cached_audio
                else:
                    log.debug("No cache hit, trying daemon pool...")
                    # 2. Try ultra-fast daemon pool
                    audio_data = self._synthesize_with_daemon_fast(text)
                    if audio_data:
                        elapsed = (time.time() - start_time) * 1000
                        log.info(f"Fast daemon synthesis for: {text[:30]}... ({elapsed:.1f}ms)")
                    else:
                        log.debug("Fast daemon failed, trying standard daemon...")
                        # 3. Fallback to original daemon (only if not using pool)
                        if not getattr(self, '_use_daemon_pool', False):
                            if self._piper_process and self._piper_process.poll() is None:
                                audio_data = self._synthesize_with_daemon(text)
                                if audio_data:
                                    elapsed = (time.time() - start_time) * 1000
                                    log.info(f"Standard daemon synthesis for: {text[:30]}... ({elapsed:.1f}ms)")
                            else:
                                log.debug("No standard daemon available")
                        else:
                            log.debug("Skipping standard daemon (using pool mode)")
                        
                        # 4. Final fallback to subprocess
                        if not audio_data:
                            log.warning(f"All daemon methods failed, using subprocess fallback")
                            audio_data = self._synthesize_subprocess(text)
                            if audio_data:
                                elapsed = (time.time() - start_time) * 1000
                                log.info(f"Subprocess fallback for: {text[:30]}... ({elapsed:.1f}ms)")
                
                # Play audio if we're still speaking
                if audio_data and self._speaking and self._shared_player:
                    try:
                        if index is not None:
                            synthIndexReached.notify(synth=self, index=index)
                        
                        # Feed audio immediately for low latency - no sync needed
                        self._shared_player.feed(audio_data)
                        
                        # Small delay to prevent synthesis from getting too far ahead
                        # This allows audio to start playing while next chunk synthesizes
                        time.sleep(0.05)  # 50ms delay for optimal streaming
                        
                    except Exception as e:
                        log.warning(f"Playback error (possible audio session disconnect): {e}")
                        # Reset audio player and try to continue
                        self._reset_audio_player()
                        # Try once more with new player
                        try:
                            with self._shared_player_lock:
                                if not self._shared_player:
                                    self._shared_player = nvwave.WavePlayer(
                                        channels=1,
                                        samplesPerSec=22050,
                                        bitsPerSample=16
                                    )
                                if self._shared_player and self._speaking:
                                    self._shared_player.feed(audio_data)
                        except Exception as e2:
                            log.error(f"Failed to recover audio player: {e2}")
                            # Continue without audio for this chunk
                        
            except queue.Empty:
                continue  # Timeout, check if still running
            except Exception as e:
                log.debug(f"Synthesis worker error: {e}")
        
        log.debug("Synthesis worker thread stopped")

    def _start_daemon_pool(self):
        """Start pool of daemon processes for parallel synthesis"""
        def start_pool():
            try:
                log.info(f"Starting optimized daemon pool with {self._max_daemons} processes")
                successful_daemons = 0
                for i in range(self._max_daemons):
                    log.debug(f"Creating daemon {i+1}/{self._max_daemons}...")
                    daemon_process = self._create_daemon_process()
                    if daemon_process:
                        with self._daemon_pool_lock:
                            self._daemon_pool.append(daemon_process)
                        successful_daemons += 1
                        log.info(f"Daemon {i+1}/{self._max_daemons} started successfully (PID: {daemon_process.pid})")
                    else:
                        log.warning(f"Failed to start daemon {i+1}, continuing with fewer daemons")
                
                with self._daemon_pool_lock:
                    if successful_daemons > 0:
                        log.info(f"Daemon pool ready: {len(self._daemon_pool)}/{self._max_daemons} daemons running - TTS operational")
                    else:
                        log.error("No daemons started successfully - TTS may not work properly")
                    
            except Exception as e:
                log.error(f"Failed to start daemon pool: {e}")
        
        threading.Thread(target=start_pool, daemon=True).start()

    def _start_daemon_monitor(self):
        """Start enhanced daemon health monitoring with crash recovery"""
        def monitor_daemons():
            consecutive_failures = 0
            last_failure_time = 0
            
            while self._synthesis_thread_running:
                try:
                    time.sleep(3.0)  # Check every 3 seconds for faster recovery
                    current_time = time.time()
                    
                    with self._daemon_pool_lock:
                        initial_count = len(self._daemon_pool)
                        # Remove dead daemons and log their exit codes
                        alive_daemons = []
                        for d in self._daemon_pool:
                            if d and d.poll() is None:
                                alive_daemons.append(d)
                            elif d:
                                exit_code = d.poll()
                                log.warning(f"Daemon died with exit code: {exit_code}")
                        
                        self._daemon_pool = alive_daemons
                        dead_count = initial_count - len(self._daemon_pool)
                        
                        if dead_count > 0:
                            log.warning(f"Detected {dead_count} dead daemons (DETACHED_PROCESS isolation), restarting...")
                            # Track failures for potential cmd window crash detection
                            if current_time - last_failure_time < 10.0:  # Multiple failures within 10 seconds
                                consecutive_failures += 1
                                if consecutive_failures >= 3:
                                    log.error(f"Multiple daemon failures detected - possible external termination (cmd window crash)")
                            else:
                                consecutive_failures = 1
                            last_failure_time = current_time
                            
                        # Restart dead daemons with enhanced retry logic
                        attempts = 0
                        while len(self._daemon_pool) < self._max_daemons and attempts < 5:
                            new_daemon = self._create_daemon_process()
                            if new_daemon:
                                self._daemon_pool.append(new_daemon)
                                log.info(f"Restarted daemon (PID: {new_daemon.pid}), pool size now: {len(self._daemon_pool)}")
                                consecutive_failures = 0  # Reset failure counter on success
                            else:
                                attempts += 1
                                if attempts < 5:
                                    time.sleep(0.5)  # Brief delay before retry
                                else:
                                    log.error(f"Failed to restart daemon after {attempts} attempts")
                                    break
                                    
                except Exception as e:
                    log.debug(f"Daemon monitor error: {e}")
                    
        threading.Thread(target=monitor_daemons, daemon=True).start()

    def _start_audio_session_monitor(self):
        """Monitor for audio session disconnections that indicate cmd window crashes"""
        def monitor_audio_sessions():
            last_synthesis_time = time.time()
            consecutive_failures = 0
            
            while self._synthesis_thread_running:
                try:
                    time.sleep(1.0)  # Check every second for audio issues
                    current_time = time.time()
                    
                    # Check if we've had recent synthesis activity
                    if hasattr(self, '_last_synthesis_time'):
                        time_since_synthesis = current_time - self._last_synthesis_time
                        
                        # If synthesis was recent but all daemons are dead, this might be a crash
                        if time_since_synthesis < 30.0:  # Recent activity within 30 seconds
                            with self._daemon_pool_lock:
                                alive_daemons = [d for d in self._daemon_pool if d and d.poll() is None]
                                if len(alive_daemons) == 0 and len(self._daemon_pool) > 0:
                                    consecutive_failures += 1
                                    log.warning(f"All daemons died after recent synthesis - possible audio session crash (failure #{consecutive_failures})")
                                    
                                    if consecutive_failures >= 2:
                                        log.error("Multiple daemon failures detected - implementing emergency restart")
                                        # Emergency restart of entire daemon pool
                                        self._emergency_restart_daemon_pool()
                                        consecutive_failures = 0
                                else:
                                    consecutive_failures = 0
                    
                    # Check for audio player issues
                    try:
                        with self._shared_player_lock:
                            if self._shared_player:
                                # Try to check if player is still functional
                                # If this fails, the audio session might be disconnected
                                pass
                    except Exception as audio_error:
                        log.warning(f"Audio player check failed: {audio_error}")
                        # Reset audio player
                        self._reset_audio_player()
                        
                except Exception as e:
                    log.debug(f"Audio session monitor error: {e}")
                    
        threading.Thread(target=monitor_audio_sessions, daemon=True).start()

    def _emergency_restart_daemon_pool(self):
        """Emergency restart of the entire daemon pool"""
        try:
            log.info("Starting emergency restart of daemon pool...")
            
            with self._daemon_pool_lock:
                # Forcefully terminate all existing daemons
                for daemon in self._daemon_pool:
                    try:
                        if daemon:
                            daemon.kill()
                    except:
                        pass
                
                # Clear the pool
                self._daemon_pool.clear()
                
                # Wait a moment for cleanup
                time.sleep(0.5)
                
                # Restart daemon pool
                for i in range(self._max_daemons):
                    new_daemon = self._create_daemon_process()
                    if new_daemon:
                        self._daemon_pool.append(new_daemon)
                        log.info(f"Emergency restart: Created daemon {i+1}/{self._max_daemons} (PID: {new_daemon.pid})")
                    else:
                        log.error(f"Emergency restart: Failed to create daemon {i+1}")
                        break
                
                log.info(f"Emergency restart complete: {len(self._daemon_pool)}/{self._max_daemons} daemons running")
                
        except Exception as e:
            log.error(f"Emergency restart failed: {e}")

    def _reset_audio_player(self):
        """Reset the shared audio player in case of audio session issues"""
        try:
            with self._shared_player_lock:
                if self._shared_player:
                    try:
                        self._shared_player.close()
                    except:
                        pass
                    self._shared_player = None
                    log.info("Reset audio player due to potential session disconnect")
        except Exception as e:
            log.debug(f"Audio player reset error: {e}")

    def _create_daemon_process(self):
        """Create a single daemon process with maximum isolation"""
        return self._create_daemon_process_direct()

    def _create_daemon_process_direct(self):
        """Direct daemon creation with ultimate isolation"""
        try:
            if not ADDON_DIR:
                log.debug("ADDON_DIR is None, cannot create daemon")
                return None
                
            addon_path = os.path.abspath(ADDON_DIR)
            piper_exe = os.path.join(addon_path, "piper", "piper.exe")
            
            log.debug(f"Looking for Piper exe at: {piper_exe}")
            if not os.path.exists(piper_exe):
                log.debug(f"Piper exe not found at: {piper_exe}")
                return None
            
            model_path = self._get_model_path()
            if not model_path:
                log.debug("Model path not found")
                return None
            
            log.debug(f"Using model: {model_path}")
            
            # Direct subprocess creation with maximum isolation
            cmd = [piper_exe, "--model", model_path, "--daemon"]
            log.debug(f"Starting daemon with command: {' '.join(cmd)}")
            
            startupinfo = None
            creation_flags = 0
            if hasattr(subprocess, 'STARTUPINFO'):
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
                # Every possible isolation flag for Windows
                creation_flags = 0
                flags_used = []
                if hasattr(subprocess, 'DETACHED_PROCESS'):
                    creation_flags |= subprocess.DETACHED_PROCESS
                    flags_used.append('DETACHED_PROCESS')
                if hasattr(subprocess, 'CREATE_NEW_PROCESS_GROUP'):
                    creation_flags |= subprocess.CREATE_NEW_PROCESS_GROUP
                    flags_used.append('CREATE_NEW_PROCESS_GROUP')
                if hasattr(subprocess, 'CREATE_NO_WINDOW'):
                    creation_flags |= subprocess.CREATE_NO_WINDOW
                    flags_used.append('CREATE_NO_WINDOW')
                if hasattr(subprocess, 'CREATE_BREAKAWAY_FROM_JOB'):
                    creation_flags |= subprocess.CREATE_BREAKAWAY_FROM_JOB
                    flags_used.append('CREATE_BREAKAWAY_FROM_JOB')
                if hasattr(subprocess, 'CREATE_DEFAULT_ERROR_MODE'):
                    creation_flags |= subprocess.CREATE_DEFAULT_ERROR_MODE
                    flags_used.append('CREATE_DEFAULT_ERROR_MODE')
                
                log.info(f"Using Windows isolation flags: {', '.join(flags_used)}")
            
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                bufsize=0,
                cwd=os.path.join(addon_path, "piper"),
                startupinfo=startupinfo,
                creationflags=creation_flags
            )
            
            log.debug(f"Daemon process created with PID: {process.pid}")
            
            # Quick startup check
            time.sleep(0.1)  # Give it a bit more time
            if process.poll() is None:
                log.info(f"Daemon PID {process.pid} is running with isolation: {', '.join(flags_used) if 'flags_used' in locals() else 'none'}")
                return process
            else:
                log.debug(f"Daemon PID {process.pid} died with exit code: {process.returncode}")
                # Try to read stderr for error info
                try:
                    stderr_data = process.stderr.read()
                    if stderr_data:
                        stderr_text = stderr_data.decode('utf-8', errors='ignore')
                        log.debug(f"Daemon stderr: {stderr_text}")
                except:
                    pass
                return None
                
        except Exception as e:
            log.error(f"Direct daemon creation failed: {e}")
            return None



    def _get_available_daemon(self):
        """Get an available daemon from the pool"""
        with self._daemon_pool_lock:
            log.debug(f"Daemon pool status: {len(self._daemon_pool)} daemons available")
            # Find a running daemon
            for i, daemon in enumerate(self._daemon_pool):
                if daemon and daemon.poll() is None:
                    log.debug(f"Using daemon {i} from pool")
                    return daemon
                else:
                    log.debug(f"Daemon {i} is dead (poll={daemon.poll() if daemon else 'None'})")
            
            # If no daemon available, try to create one quickly
            log.debug("No available daemons, creating new one...")
            new_daemon = self._create_daemon_process()
            if new_daemon:
                self._daemon_pool.append(new_daemon)
                log.debug(f"Created new daemon, pool size now: {len(self._daemon_pool)}")
                return new_daemon
            else:
                log.warning("Failed to create new daemon")
            
            return None

    def _pre_cache_common_words(self):
        """Pre-synthesize common words for instant playback"""
        common_words = [
            "button", "edit", "link", "list", "menu", "window", "dialog", "text",
            "selected", "checked", "unchecked", "expanded", "collapsed", "loading",
            "page", "document", "heading", "paragraph", "table", "row", "column",
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"
        ]
        
        def cache_words():
            try:
                for word in common_words:
                    if not self._speaking:  # Don't cache if we should be quiet
                        break
                    audio_data = self._synthesize_with_daemon_fast(word)
                    if audio_data:
                        with self._cache_lock:
                            cache_key = f"{word.lower()}|rate_{self._rate}"
                            self._audio_cache[cache_key] = audio_data
                        log.debug(f"Cached audio for: {word} at rate {self._rate}")
                    time.sleep(0.01)  # Small delay between caching
            except Exception as e:
                log.debug(f"Word caching error: {e}")
        
        threading.Thread(target=cache_words, daemon=True).start()

    def _get_cached_audio(self, text: str) -> Optional[bytes]:
        """Get cached audio if available"""
        clean_text = text.lower().strip()
        # Include rate in cache key so different rates don't share cached audio
        cache_key = f"{clean_text}|rate_{self._rate}"
        with self._cache_lock:
            return self._audio_cache.get(cache_key)

    def _synthesize_with_daemon_fast(self, text: str) -> Optional[bytes]:
        """Ultra-fast daemon synthesis optimized for <10ms"""
        log.debug(f"Fast synthesis called for: {text[:30]}...")
        
        # Try cache first for instant response
        cached_audio = self._get_cached_audio(text)
        if cached_audio:
            log.debug(f"Cache hit for: {text[:20]}...")
            return cached_audio
        
        # Get daemon from pool
        log.debug("Getting daemon from pool...")
        daemon = self._get_available_daemon()
        if not daemon:
            log.debug("No daemon available from pool")
            return None
        
        log.debug(f"Got daemon from pool, attempting synthesis...")
        
        try:
            # Minimal JSON for speed
            clean_text = text.strip()[:200]  # Limit length for speed
            # Convert rate (0-100) to speed (1-10) for Piper daemon
            speed = max(1, min(10, int((self._rate / 10) + 1)))
            
            request_json = f'{{"text":"{clean_text}","speed":{speed}}}\n'
            log.debug(f"Sending fast request: {request_json.strip()} (rate={self._rate})")
            
            # Send request
            try:
                daemon.stdin.write(request_json.encode('utf-8'))
                daemon.stdin.flush()
                log.debug("Fast request sent successfully")
            except Exception as send_error:
                log.error(f"Failed to send fast request: {send_error}")
                return None
            
            # Check daemon is still alive
            if daemon.poll() is not None:
                log.error(f"Daemon died after sending request (exit code: {daemon.returncode})")
                return None
            
            # Fast response reading with proper chunked reading
            start_time = time.time()
            log.debug("Starting fast response read...")
            
            # Read size header first
            size_data = None
            while time.time() - start_time < 2.0:  # Increase timeout for reading
                try:
                    size_data = daemon.stdout.read(4)
                    if size_data and len(size_data) == 4:
                        break
                    elif size_data:
                        log.debug(f"Incomplete size data: {len(size_data)} bytes")
                except Exception as read_error:
                    log.debug(f"Size read error: {read_error}")
                time.sleep(0.001)
            
            if not size_data or len(size_data) != 4:
                elapsed = (time.time() - start_time) * 1000
                log.warning(f"Failed to read size header after {elapsed:.1f}ms")
                return None
            
            audio_size = struct.unpack('<I', size_data)[0]
            log.debug(f"Fast daemon returned audio size: {audio_size}")
            
            if not (0 < audio_size < 1024*1024):  # Reasonable size check
                log.debug(f"Invalid audio size: {audio_size}")
                return None
            
            # Read audio data in chunks to ensure we get all bytes
            audio_data = b''
            bytes_remaining = audio_size
            read_start = time.time()
            
            while bytes_remaining > 0 and time.time() - read_start < 3.0:  # 3 second timeout for audio data
                try:
                    chunk_size = min(bytes_remaining, 8192)  # Read in 8KB chunks
                    chunk = daemon.stdout.read(chunk_size)
                    if not chunk:
                        # No data available, wait a bit
                        time.sleep(0.001)
                        continue
                    
                    audio_data += chunk
                    bytes_remaining -= len(chunk)
                    log.debug(f"Read {len(chunk)} bytes, {bytes_remaining} remaining")
                    
                except Exception as read_error:
                    log.debug(f"Audio read error: {read_error}")
                    break
            
            if len(audio_data) == audio_size:
                elapsed = (time.time() - start_time) * 1000
                log.info(f"Fast daemon synthesis successful: {len(audio_data)} bytes in {elapsed:.1f}ms")
                # Cache result for future use (include rate in cache key)
                with self._cache_lock:
                    if len(self._audio_cache) < self._max_cache_size:
                        cache_key = f"{text.lower().strip()}|rate_{self._rate}"
                        self._audio_cache[cache_key] = audio_data
                return audio_data
            else:
                log.warning(f"Audio data size mismatch: got {len(audio_data)}, expected {audio_size}")
            
            elapsed = (time.time() - start_time) * 1000
            log.warning(f"Fast daemon synthesis timeout after {elapsed:.1f}ms")
            return None
            
        except Exception as e:
            log.error(f"Fast daemon synthesis error: {e}")
            return None

    def _get_shared_player(self):
        """Get the shared player instance"""
        with self._shared_player_lock:
            if not self._shared_player:
                try:
                    self._shared_player = nvwave.WavePlayer(
                        channels=1,
                        samplesPerSec=22050,
                        bitsPerSample=16
                    )
                except Exception as e:
                    log.error(f"Failed to create shared player: {e}")
                    return None
            return self._shared_player

    @classmethod  
    def check(cls):
        """Check if Piper TTS is available"""
        try:
            # Get addon directory
            addon = addonHandler.getCodeAddon()
            addon_dir = addon.path if addon else None
            
            if not addon_dir:
                return False
                
            addon_path = os.path.abspath(addon_dir)
            
            # Look for model files
            possible_dirs = [
                os.path.join(addon_path, "models"),
                os.path.join(addon_path, "piper", "models"), 
                os.path.join(addon_path, "voices"),
            ]
            
            model_file = "en_us-jane-medium.onnx"
            config_file = "en_us-jane-medium.onnx.json"
            
            for model_dir in possible_dirs:
                model_path = os.path.join(model_dir, model_file)
                config_path = os.path.join(model_dir, config_file)
                
                if os.path.exists(model_path) and os.path.exists(config_path):
                    log.info("Piper TTS is available")
                    return True
                    
            log.debug("Piper model files not found")
            return False
                
        except Exception as e:
            log.debug(f"Piper check failed: {e}")
            return False

    def _get_model_path(self) -> Optional[str]:
        """Get path to voice model files"""
        if not ADDON_DIR:
            return None
            
        addon_path = os.path.abspath(ADDON_DIR)
        
        # Look for model files
        possible_dirs = [
            os.path.join(addon_path, "models"),
            os.path.join(addon_path, "piper", "models"), 
            os.path.join(addon_path, "voices"),
        ]
        
        for model_dir in possible_dirs:
            model_file = os.path.join(model_dir, self._MODEL_FILE)
            config_file = os.path.join(model_dir, self._CONFIG_FILE)
            
            if os.path.exists(model_file) and os.path.exists(config_file):
                return model_file
                
        return None

    def _synthesize_to_wav_file(self, text: str) -> Optional[str]:
        """Synthesize text to a temporary WAV file"""
        try:
            # Try to use Python Piper implementation first
            try:
                # Add piper module to path - try multiple possible locations
                piper_paths = []
                if ADDON_DIR:
                    piper_paths.extend([
                        os.path.join(ADDON_DIR, "piper_src", "src", "python_run"),
                        os.path.join(ADDON_DIR, "piper", "src", "python_run"),
                        os.path.join(ADDON_DIR, "deps", "piper"),
                        ADDON_DIR  # Also try the addon root
                    ])
                
                # Try to add any existing piper path
                for piper_path in piper_paths:
                    if os.path.exists(piper_path) and piper_path not in sys.path:
                        sys.path.insert(0, piper_path)
                        log.debug(f"Added {piper_path} to Python path")
                        
                from piper import PiperVoice
                log.debug("Successfully imported PiperVoice")
                
                # Load voice if not already loaded
                if self._piper_voice is None:
                    model_path = self._get_model_path()
                    if model_path:
                        log.debug(f"Loading Piper voice model: {model_path}")
                        self._piper_voice = PiperVoice.load(model_path)
                        log.info("Piper voice loaded successfully")
                    else:
                        raise RuntimeError("Model path not found")
                
                # Create temporary WAV file
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_file.close()
                
                # Convert rate to length_scale for Piper
                length_scale = 2.0 - (self._rate / 100.0 * 1.5)
                length_scale = max(0.1, min(3.0, length_scale))
                
                # Synthesize to WAV file
                import wave
                with wave.open(temp_file.name, "wb") as wav_file:
                    self._piper_voice.synthesize(
                        text, 
                        wav_file,
                        length_scale=length_scale
                    )
                
                return temp_file.name
                
            except ImportError as e:
                log.warning(f"Python Piper not available: {e}. Trying fallback approaches.")
                return self._synthesize_fallback(text)
                
        except Exception as e:
            log.error(f"Synthesis error: {e}")
            return self._synthesize_fallback(text)

    def _synthesize_fallback(self, text: str) -> Optional[str]:
        """Fallback synthesis - try subprocess first, then create notification sound"""
        try:
            # First try subprocess if executable exists
            if ADDON_DIR:
                addon_path = os.path.abspath(ADDON_DIR)
                piper_exe_paths = [
                    os.path.join(addon_path, "piper", "piper.exe"),
                    os.path.join(addon_path, "bin", "piper.exe"),
                    os.path.join(addon_path, "piper.exe"),
                ]
                
                piper_exe = None
                for path in piper_exe_paths:
                    if os.path.exists(path):
                        piper_exe = path
                        break
                        
                if piper_exe:
                    model_path = self._get_model_path()
                    if model_path:
                        try:
                            # Create temporary WAV file
                            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                            temp_file.close()
                            
                            # Run piper subprocess
                            cmd = [
                                piper_exe,
                                "--model", model_path,
                                "--output_file", temp_file.name
                            ]
                            
                            # Hide console window on Windows
                            startupinfo = None
                            if hasattr(subprocess, 'STARTUPINFO'):
                                startupinfo = subprocess.STARTUPINFO()
                                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                                startupinfo.wShowWindow = subprocess.SW_HIDE
                            
                            process = subprocess.Popen(
                                cmd,
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                startupinfo=startupinfo  # Hide console window
                            )
                            
                            stdout, stderr = process.communicate(input=text, timeout=30)
                            
                            if process.returncode == 0 and os.path.exists(temp_file.name):
                                log.info("Subprocess synthesis successful")
                                return temp_file.name
                            else:
                                log.warning(f"Piper subprocess failed: {stderr}")
                        except Exception as e:
                            log.warning(f"Subprocess synthesis error: {e}")
            
            # If we get here, both Python Piper and subprocess failed
            # Create a simple notification to indicate that Piper TTS is not fully functional
            log.warning("Piper TTS not available - neither Python module nor executable found")
            log.info(f"Attempted to synthesize: {text[:50]}{'...' if len(text) > 50 else ''}")
            
            # Return None to indicate synthesis failure - NVDA will handle this gracefully
            return None
                
        except Exception as e:
            log.error(f"Fallback synthesis error: {e}")
            return None

    def _start_piper_daemon(self):
        """Start new Piper process for synthesis"""
        with self._daemon_lock:
            # Double-check inside the lock
            if self._piper_process and self._piper_process.poll() is None:
                # log.debug("Piper daemon already running, reusing existing process")
                return True  # Already running
                
            # Clean up any existing dead process
            if self._piper_process:
                try:
                    self._piper_process.terminate()
                except:
                    try:
                        self._piper_process.kill()
                    except:
                        pass
                self._piper_process = None
                
            try:
                # Find piper executable
                if not ADDON_DIR:
                    log.error("ADDON_DIR is None")
                    return False
                    
                addon_path = os.path.abspath(ADDON_DIR)
                piper_exe = os.path.join(addon_path, "piper", "piper.exe")
                
                log.debug(f"Looking for Piper executable at: {piper_exe}")
                if not os.path.exists(piper_exe):
                    log.error(f"Piper executable not found at: {piper_exe}")
                    return False
                else:
                    log.debug(f"Piper executable found at: {piper_exe}")
                
                model_path = self._get_model_path()
                if not model_path:
                    log.error("Model path not found")
                    return False
                
                # Start Piper in daemon mode with optimized settings and correct espeak data path
                espeak_data_path = os.path.join(addon_path, "piper", "espeak-ng-data").replace('/', '\\')
                
                # Check if espeak data path exists
                if not os.path.exists(espeak_data_path):
                    log.warning(f"eSpeak data path not found: {espeak_data_path}")
                    # Try without espeak_data parameter
                    cmd = [
                        piper_exe,
                        "--model", model_path,
                        "--daemon"
                    ]
                else:
                    cmd = [
                        piper_exe,
                        "--model", model_path,
                        "--daemon",
                        "--espeak_data", espeak_data_path
                    ]
                # Note: C++ optimizations (ONNX, tensor pooling, async phonemization) are automatically enabled
                
                log.info(f"Starting NEW Piper daemon: {' '.join(cmd)}")
                
                # Hide console window and detach from console on Windows
                startupinfo = None
                creation_flags = 0
                if hasattr(subprocess, 'STARTUPINFO'):
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    startupinfo.wShowWindow = subprocess.SW_HIDE
                    # ULTIMATE isolation: DETACHED_PROCESS completely detaches from console
                    creation_flags = 0
                    if hasattr(subprocess, 'DETACHED_PROCESS'):
                        creation_flags |= subprocess.DETACHED_PROCESS
                        log.debug("Using DETACHED_PROCESS for complete console isolation")
                    else:
                        # Fallback isolation techniques if DETACHED_PROCESS not available
                        if hasattr(subprocess, 'CREATE_NEW_PROCESS_GROUP'):
                            creation_flags |= subprocess.CREATE_NEW_PROCESS_GROUP
                        if hasattr(subprocess, 'CREATE_NO_WINDOW'):
                            creation_flags |= subprocess.CREATE_NO_WINDOW
                        if hasattr(subprocess, 'CREATE_BREAKAWAY_FROM_JOB'):
                            creation_flags |= subprocess.CREATE_BREAKAWAY_FROM_JOB
                    # Force process to run in background
                    if hasattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS'):
                        creation_flags |= subprocess.BELOW_NORMAL_PRIORITY_CLASS
                
                self._piper_process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=False,  # Binary mode for audio data
                    bufsize=0,
                    cwd=os.path.join(addon_path, "piper"),  # Set working directory to piper folder
                    startupinfo=startupinfo,  # Hide console window
                    creationflags=creation_flags  # Detach from console - prevents NVDA crashes!
                )
                
                log.debug(f"Daemon process created with PID: {self._piper_process.pid}")
                
                # Give daemon a moment to initialize
                time.sleep(0.1)
                
                # Check if daemon started successfully
                if self._piper_process.poll() is not None:
                    # Process died immediately
                    stderr_output = ""
                    try:
                        stderr_data = self._piper_process.stderr.read()
                        if stderr_data:
                            stderr_output = stderr_data.decode('utf-8', errors='ignore')
                    except:
                        pass
                    log.error(f"Piper daemon died immediately (exit code: {self._piper_process.returncode}), stderr: {stderr_output}")
                    self._piper_process = None
                    return False
                
                log.info(f"Piper daemon started successfully with PID: {self._piper_process.pid}")
                self._reset_daemon_cleanup_timer()
                return True
                
            except Exception as e:
                log.error(f"Failed to start Piper daemon: {e}")
                self._piper_process = None
                return False

    def _stop_piper_daemon(self):
        """Stop Piper daemon process"""
        with self._daemon_lock:
            if self._piper_process:
                try:
                    self._piper_process.terminate()
                    self._piper_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._piper_process.kill()
                    self._piper_process.wait()
                except Exception as e:
                    log.debug(f"Error stopping Piper daemon: {e}")
                finally:
                    self._piper_process = None
                    self._cancel_daemon_cleanup_timer()
                    log.info("Piper daemon stopped")
                
    def _synthesize_with_daemon(self, text: str) -> Optional[bytes]:
        """Ultra-fast daemon synthesis for <50ms latency"""
        log.debug(f"_synthesize_with_daemon called with text: {text[:50]}...")
        
        # Skip locking for maximum speed - risk of race conditions but prioritize latency
        # Only check if speaking is still active
        if not self._speaking:
            log.debug("Daemon synthesis skipped - not speaking")
            return None
            
        log.debug("Checking daemon availability...")
        try:
            if not self._start_piper_daemon():
                log.error("Failed to start Piper daemon in _synthesize_with_daemon")
                return None  # Quick fail for speed
                
            # REMOVE daemon lock for ultra-low latency - accept race condition risk
            # Check daemon is still running
            log.debug(f"Daemon process check: process={self._piper_process}, poll={self._piper_process.poll() if self._piper_process else 'None'}")
            if not self._piper_process or self._piper_process.poll() is not None:
                log.warning("Piper daemon died, attempting restart")
                self._piper_process = None
                if not self._start_piper_daemon():
                    log.error("Failed to restart daemon")
                    return None
                
            log.debug("Daemon confirmed running, starting synthesis...")
            try:
                # Convert rate (0-100) to speed (1-10) for Piper daemon
                # rate 0-20 -> speed 1, rate 21-40 -> speed 2, etc.
                speed = max(1, min(10, int((self._rate / 10) + 1)))
                
                # Prepare JSON request as expected by daemon mode
                # Clean the text to ensure valid JSON (preserve single characters)
                clean_text = text.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')
                clean_text = ''.join(c for c in clean_text if ord(c) >= 32 or c in '\n\r\t')
                clean_text = clean_text.strip()
                
                if not clean_text:
                    log.debug(f"Empty text after cleaning original: '{text}'")
                    return None
                
                request = {
                    "text": clean_text,
                    "speed": speed
                }
                
                try:
                    request_json = json.dumps(request, ensure_ascii=True) + "\n"
                    log.debug(f"Sending daemon request: {request_json.strip()}")
                except (TypeError, ValueError) as json_error:
                    log.error(f"Failed to create JSON: {json_error}")
                    return None
                
                # Send JSON to Piper daemon
                try:
                    self._piper_process.stdin.write(request_json.encode('utf-8'))
                    self._piper_process.stdin.flush()
                    log.debug("JSON sent successfully to daemon")
                except (BrokenPipeError, OSError) as pipe_error:
                    log.error(f"Failed to send to daemon: {pipe_error}")
                    raise RuntimeError("Daemon pipe broken")
                
                # Check if daemon process has died
                if self._piper_process.poll() is not None:
                    # Process died, read stderr for error
                    try:
                        stderr_data = self._piper_process.stderr.read()
                        if stderr_data:
                            log.error(f"Piper daemon died with error: {stderr_data.decode('utf-8', errors='ignore')}")
                    except:
                        pass
                    raise RuntimeError("Piper daemon process died")
                
                # Read response size (4 bytes) with timeout
                import select
                import time
                
                log.debug("Waiting for daemon response...")
                
                # Check if data is available with timeout
                start_time = time.time()
                size_data = None
                while time.time() - start_time < 5.0:  # Increase timeout to 5 seconds for debugging
                    # Check if speech has been cancelled - abort quickly
                    if not self._speaking:
                        log.debug("Speech cancelled while waiting for daemon response, aborting")
                        return None
                    
                    if self._piper_process.poll() is not None:
                        # Process died, try to read stderr for diagnostic info
                        daemon_exit_code = self._piper_process.returncode
                        stderr_output = ""
                        try:
                            stderr_data = self._piper_process.stderr.read()
                            if stderr_data:
                                stderr_output = stderr_data.decode('utf-8', errors='ignore')
                                log.error(f"Daemon died (exit code {daemon_exit_code}) with stderr: {stderr_output}")
                            else:
                                log.error(f"Daemon died (exit code {daemon_exit_code}) with no stderr output")
                        except Exception as stderr_error:
                            log.error(f"Daemon died (exit code {daemon_exit_code}), failed to read stderr: {stderr_error}")
                        
                        # Log additional diagnostic info
                        log.error(f"Daemon crash details: PID {self._piper_process.pid}, command: {' '.join(cmd) if 'cmd' in locals() else 'unknown'}")
                        raise RuntimeError(f"Piper daemon process died during read (exit code: {daemon_exit_code})")
                    
                    # Try to read available data
                    try:
                        size_data = self._piper_process.stdout.read(4)
                        if size_data:
                            log.debug(f"Successfully read {len(size_data)} bytes for size header")
                            break
                        else:
                            log.debug("No data available yet, retrying...")
                    except Exception as read_error:
                        log.debug(f"Read attempt failed: {read_error}")
                    
                    time.sleep(0.1)
                    continue
                else:
                    elapsed = time.time() - start_time
                    log.error(f"Timeout waiting for daemon response after {elapsed:.2f} seconds - daemon may be hung")
                    
                    # Try to read stderr for any error messages
                    try:
                        stderr_data = self._piper_process.stderr.read()
                        if stderr_data:
                            stderr_output = stderr_data.decode('utf-8', errors='ignore')
                            log.error(f"Daemon stderr during timeout: {stderr_output}")
                    except:
                        pass
                        
                    raise RuntimeError("Timeout waiting for daemon response")
                
                if not size_data or len(size_data) != 4:
                    raise RuntimeError(f"Failed to read audio size from Piper daemon (got {len(size_data) if size_data else 0} bytes)")
                
                audio_size = struct.unpack('<I', size_data)[0]
                log.debug(f"Received audio size from daemon: {audio_size} bytes")
                
                # Check for error response (size = 0)
                if audio_size == 0:
                    log.warning("Daemon returned empty response (size=0)")
                    return None
                
                # Read audio data
                if audio_size > 0:
                    if audio_size > 10 * 1024 * 1024:  # Sanity check: max 10MB
                        raise RuntimeError(f"Unreasonable audio size: {audio_size} bytes")
                    
                    # Read audio data in chunks until we have all bytes
                    audio_data = b''
                    bytes_remaining = audio_size
                    
                    while bytes_remaining > 0:
                        # Check if speech has been cancelled - abort quickly
                        if not self._speaking:
                            log.debug("Speech cancelled while reading audio data, aborting")
                            return None
                        
                        chunk = self._piper_process.stdout.read(bytes_remaining)
                        if not chunk:
                            raise RuntimeError(f"Daemon stream ended prematurely (got {len(audio_data)}/{audio_size} bytes)")
                        audio_data += chunk
                        bytes_remaining -= len(chunk)
                    
                    # Reset cleanup timer since daemon is actively being used
                    self._reset_daemon_cleanup_timer()
                    return audio_data
                else:
                    log.debug("Daemon returned empty audio data")
                    return None
                
            except Exception as e:
                log.error(f"Piper daemon synthesis error: {e}")
                # Mark daemon as failed so other threads will restart it
                log.warning("Marking daemon as failed due to error")
                if self._piper_process:
                    try:
                        # Don't wait too long for process termination to avoid blocking
                        self._piper_process.terminate()
                    except:
                        try:
                            self._piper_process.kill()
                        except:
                            pass
                    self._piper_process = None
                
                return None
        except Exception as e:
            log.error(f"Daemon synthesis error: {e}")
            # Try to clean up failed daemon and restart daemon pool
            try:
                if hasattr(self, '_daemon_pool') and hasattr(self, '_daemon_pool_lock'):
                    with self._daemon_pool_lock:
                        # Remove dead daemons from pool
                        self._daemon_pool = [d for d in self._daemon_pool if d and d.poll() is None]
                        # Try to restart one daemon
                        if len(self._daemon_pool) < self._max_daemons:
                            new_daemon = self._create_daemon_process()
                            if new_daemon:
                                self._daemon_pool.append(new_daemon)
                                log.debug("Restarted daemon after error")
            except:
                pass
            return None

    def _synthesize_subprocess(self, text: str) -> Optional[bytes]:
        """Synthesize text using Piper subprocess with --output_raw"""
        try:
            # Find piper executable
            if not ADDON_DIR:
                return None
                
            addon_path = os.path.abspath(ADDON_DIR)
            piper_exe = os.path.join(addon_path, "piper", "piper.exe")
            
            if not os.path.exists(piper_exe):
                log.error(f"Piper executable not found at: {piper_exe}")
                return None
            
            model_path = self._get_model_path()
            if not model_path:
                log.error("Model path not found")
                return None
            
            # Create Piper process for this synthesis
            cmd = [
                piper_exe,
                "--model", model_path,
                "--output_raw"
            ]
            
            log.debug(f"Starting Piper subprocess: {' '.join(cmd)}")
            
            # Properly detach from console and hide window on Windows
            startupinfo = None
            creation_flags = 0
            if hasattr(subprocess, 'STARTUPINFO'):
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
                # ULTIMATE isolation: DETACHED_PROCESS completely detaches from console
                creation_flags = 0
                if hasattr(subprocess, 'DETACHED_PROCESS'):
                    creation_flags |= subprocess.DETACHED_PROCESS
                    log.debug("Using DETACHED_PROCESS for complete console isolation")
                else:
                    # Fallback isolation techniques if DETACHED_PROCESS not available
                    if hasattr(subprocess, 'CREATE_NEW_PROCESS_GROUP'):
                        creation_flags |= subprocess.CREATE_NEW_PROCESS_GROUP
                    if hasattr(subprocess, 'CREATE_NO_WINDOW'):
                        creation_flags |= subprocess.CREATE_NO_WINDOW
                    if hasattr(subprocess, 'CREATE_BREAKAWAY_FROM_JOB'):
                        creation_flags |= subprocess.CREATE_BREAKAWAY_FROM_JOB
            
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,  # Binary mode for audio data
                cwd=os.path.join(addon_path, "piper"),  # Set working directory to piper folder
                startupinfo=startupinfo,  # Hide console window
                creationflags=creation_flags  # Detach from console
            )
            
            # Send text and get audio output
            text_input = text.encode('utf-8')
            
            log.debug(f"Sending text to Piper: {text[:50]}...")
            
            try:
                stdout_data, stderr_data = process.communicate(input=text_input, timeout=5.0)
                
                if stderr_data:
                    stderr_text = stderr_data.decode('utf-8', errors='ignore')
                    if stderr_text.strip():
                        log.debug(f"Piper stderr: {stderr_text}")
                
                if stdout_data:
                    log.debug(f"Successfully received {len(stdout_data)} bytes of raw PCM audio")
                    return stdout_data
                else:
                    log.debug("No audio data received from Piper")
                    return None
                    
            except subprocess.TimeoutExpired:
                log.error("Piper process timeout")
                process.kill()
                process.wait()
                return None
            except Exception as comm_error:
                log.error(f"Error communicating with Piper: {comm_error}")
                return None
                
        except Exception as e:
            log.error(f"Piper synthesis error: {e}")
            return None

    def _chunk_text_intelligently(self, text: str, max_chunk_size: int = 200) -> list:
        """Split text into optimal chunks for synthesis"""
        if len(text) <= max_chunk_size:
            return [text]
        
        import re
        chunks = []
        
        # Split by sentence boundaries first
        sentences = re.split(r'[.!?]+\s+', text)
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed chunk size, start new chunk
            if current_chunk and len(current_chunk + " " + sentence) > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        # If no sentence boundaries found, split by words
        if len(chunks) == 1 and len(chunks[0]) > max_chunk_size:
            words = chunks[0].split()
            chunks = []
            current_chunk = ""
            
            for word in words:
                if current_chunk and len(current_chunk + " " + word) > max_chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = word
                else:
                    if current_chunk:
                        current_chunk += " " + word
                    else:
                        current_chunk = word
            
            if current_chunk:
                chunks.append(current_chunk.strip())
        
        return chunks

    def _cleanup_temp_file(self, file_path: str):
        """Clean up temporary file"""
        try:
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            log.debug(f"Failed to cleanup temp file {file_path}: {e}")

    def speak(self, speechSequence: SpeechSequence):
        """Speak the given speech sequence with crash protection"""
        try:
            self._speaking = True
            self._current_index = 0
            
            # Process speech sequence to extract text and commands with intelligent chunking
            text_parts = []
            current_text = ""
        
            for item in speechSequence:
                if isinstance(item, str):
                    current_text += item
                elif isinstance(item, IndexCommand):
                    if current_text.strip():
                        # Intelligent chunking: split long text at sentence boundaries
                        chunks = self._chunk_text_intelligently(current_text.strip())
                        for chunk in chunks:
                            text_parts.append((chunk, item.index))
                        current_text = ""
                    self._current_index = item.index
                elif isinstance(item, BreakCommand):
                    current_text += " "
                elif isinstance(item, RateCommand):
                    self._rate = item.newValue
                elif isinstance(item, VolumeCommand):
                    self._volume = item.newValue
                elif isinstance(item, PitchCommand):
                    # Piper doesn't directly support pitch changes, but we acknowledge the command
                    pass
                elif isinstance(item, CharacterModeCommand):
                    # Handle character mode changes
                    pass
                elif isinstance(item, PhonemeCommand):
                    # Handle phoneme commands if needed
                    pass
                    
            # Add remaining text with chunking
            if current_text.strip():
                chunks = self._chunk_text_intelligently(current_text.strip())
                for chunk in chunks:
                    text_parts.append((chunk, None))
                
            # Clear any pending synthesis tasks
            try:
                while not self._synthesis_queue.empty():
                    self._synthesis_queue.get_nowait()
            except queue.Empty:
                pass
            
            # Queue text parts for sequential synthesis
            for text, index in text_parts:
                if not self._speaking:
                    break
                    
                # Skip only truly empty text (allow single characters for character speech)
                if not text or not text.strip():
                    continue
                
                # Add to synthesis queue for sequential processing
                self._synthesis_queue.put((text, index))
            
            # Quick done notification after queueing
            def delayed_done():
                # Wait a bit for synthesis to start, then notify done
                threading.Timer(0.5, lambda: synthDoneSpeaking.notify(synth=self) if self._speaking else None).start()
            
            threading.Thread(target=delayed_done, daemon=True).start()
            
        except Exception as e:
            log.error(f"Critical error in speak() - implementing crash protection: {e}")
            # Emergency fallback: try to continue with basic functionality
            try:
                self._speaking = False
                synthDoneSpeaking.notify(synth=self)
            except:
                pass

    def cancel(self):
        """Ultra-fast cancel for <50ms response with crash protection"""
        try:
            self._speaking = False
            
            # Clear synthesis queue
            try:
                while not self._synthesis_queue.empty():
                    self._synthesis_queue.get_nowait()
            except queue.Empty:
                pass
            
            # Stop shared player immediately
            with self._shared_player_lock:
                if self._shared_player:
                    try:
                        self._shared_player.stop()
                    except Exception as e:
                        log.debug(f"Error stopping player during cancel: {e}")
                        # Reset player if it failed
                        self._reset_audio_player()
            
            # Keep daemon alive for next synthesis (better latency)
            
        except Exception as e:
            log.error(f"Critical error in cancel() - implementing crash protection: {e}")
            # Emergency fallback
            try:
                self._speaking = False
            except:
                pass

    def pause(self, switch: bool):
        """Pause/resume speech"""
        with self._shared_player_lock:
            if self._shared_player:
                try:
                    self._shared_player.pause(switch)
                except Exception as e:
                    log.debug(f"Error pausing player: {e}")
                    # Don't reset player on pause errors

    def terminate(self):
        """Clean up when driver is terminated"""
        self.cancel()
        
        # Stop synthesis worker thread
        self._synthesis_thread_running = False
        if self._synthesis_thread:
            # Send shutdown signal
            self._synthesis_queue.put(None)
            try:
                self._synthesis_thread.join(timeout=2.0)
            except:
                pass
        
        # Close shared player
        with self._shared_player_lock:
            if self._shared_player:
                try:
                    self._shared_player.close()
                except:
                    pass
                self._shared_player = None
        
        # Clean up daemon pool
        with self._daemon_pool_lock:
            for daemon in self._daemon_pool:
                try:
                    daemon.terminate()
                    daemon.wait(timeout=1.0)
                except:
                    try:
                        daemon.kill()
                    except:
                        pass
            self._daemon_pool.clear()
        
        self._cancel_daemon_cleanup_timer()
        self._stop_piper_daemon()
        self._piper_voice = None
        
        log.info("Piper TTS driver terminated")

    # Voice properties
    @property
    def availableVoices(self):
        """Available voices"""
        return self._voices

    @property 
    def voice(self):
        """Current voice"""
        return self._current_voice
        
    @voice.setter
    def voice(self, value):
        """Set current voice"""
        if value in self._voices:
            self._current_voice = value

    # Rate property
    @property
    def rate(self):
        """Current rate (0-100)"""
        return self._rate
        
    @rate.setter
    def rate(self, value):
        """Set rate (0-100)"""
        self._rate = max(0, min(100, value))

    # Volume property  
    @property
    def volume(self):
        """Current volume (0-100)"""
        return self._volume
        
    @volume.setter
    def volume(self, value):
        """Set volume (0-100)"""
        self._volume = max(0, min(100, value))
    
    def _reset_daemon_cleanup_timer(self):
        """Reset the daemon cleanup timer"""
        self._cancel_daemon_cleanup_timer()
        self._daemon_cleanup_timer = threading.Timer(
            self._daemon_idle_timeout, 
            self._cleanup_idle_daemon
        )
        self._daemon_cleanup_timer.daemon = True
        self._daemon_cleanup_timer.start()
    
    def _cancel_daemon_cleanup_timer(self):
        """Cancel the daemon cleanup timer"""
        if self._daemon_cleanup_timer:
            self._daemon_cleanup_timer.cancel()
            self._daemon_cleanup_timer = None
    
    def _cleanup_idle_daemon(self):
        """Stop daemon after idle timeout"""
        try:
            log.debug("Stopping idle daemon after timeout")
            self._stop_piper_daemon()
        except Exception as e:
            log.debug(f"Error stopping idle daemon: {e}")