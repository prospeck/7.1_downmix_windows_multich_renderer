# 7.1_downmix_windows_multich_renderer

Install Python and download/install VB-Audio Hi-Fi Cable
Set Hi-Fi Cable as both default playback and recording device in Windows
Open properties for both → set format to 24-bit, 48000 Hz
Press Win + R → type mmsys.cpl → configure Hi-Fi Cable Input to 7.1 Surround
Open Command Prompt as Administrator
Navigate to your file location → cd Desktop (or your folder)
Run the renderer → python renderer_app.py
In the app, select input → CABLE Output (VB-Audio Hi-Fi Cable) | WASAPI | IN:16 OUT:0 | 48k
Select output → your DAC / audio device (e.g., Qudelix-5K)
Click Start and play music
