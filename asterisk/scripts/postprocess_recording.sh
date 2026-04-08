#!/bin/sh

REC_WAV="$1"
REC_MP3="$2"
REC_READY="$3"

sleep 2

ffmpeg -y -i "$REC_WAV" -codec:a libmp3lame -q:a 2 "$REC_MP3" >> /var/log/asterisk/ffmpeg_ready.log 2>&1

if [ -f "$REC_MP3" ]; then
  touch "$REC_READY"
fi
