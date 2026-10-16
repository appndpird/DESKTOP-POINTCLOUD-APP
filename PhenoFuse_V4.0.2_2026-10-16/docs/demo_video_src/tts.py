r"""narration.json -> tts\<step>.wav (22050 Hz, 16-bit mono) with Windows text-to-speech."""
import json, os, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); NAR = json.load(open(os.path.join(HERE, "narration.json"), encoding="utf-8"))
out = os.path.join(HERE, "tts"); os.makedirs(out, exist_ok=True)
items = [(k, v[2]) for k, v in NAR.items()]
spec = os.path.join(out, "_spec.json"); json.dump(items, open(spec, "w", encoding="utf-8"))
ps = r'''
Add-Type -AssemblyName System.Speech
$items = Get-Content -Raw -Encoding UTF8 "%s" | ConvertFrom-Json
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voices = $s.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }
$pick = @("Microsoft Zira Desktop", "Microsoft Hazel Desktop", "Microsoft David Desktop") | Where-Object { $voices -contains $_ } | Select-Object -First 1
if ($pick) { $s.SelectVoice($pick) }
$s.Rate = 0
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(22050, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
foreach ($it in $items) { $p = Join-Path "%s" ($it[0] + ".wav"); $s.SetOutputToWaveFile($p, $fmt); $s.Speak($it[1]); $s.SetOutputToNull(); Write-Output ("" + $it[0] + " " + $pick) }
''' % (spec, out)
subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
