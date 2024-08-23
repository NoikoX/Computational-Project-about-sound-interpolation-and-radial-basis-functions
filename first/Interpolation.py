from gtts import gTTS
import librosa
import numpy as np
import plotly.graph_objects as go
from scipy.io.wavfile import write
from scipy.interpolate import CubicSpline, interp1d

text = "おまえはもう死んでる, なに?!"
speech = gTTS(text=text, lang='ja')
speech.save("test.mp3")
audio_data, sampling_rate = librosa.load("test.mp3")

frame_length = 0.02
samples_per_frame = int(frame_length * sampling_rate)
frames = np.array([audio_data[i:i + samples_per_frame] for i in range(0, len(audio_data), samples_per_frame) if
                   len(audio_data[i:i + samples_per_frame]) == samples_per_frame])
time = np.arange(0, len(audio_data)) / sampling_rate

fig = go.Figure()
for i, frame in enumerate(frames):
    fig.add_trace(go.Scatter(x=time[i * samples_per_frame:(i + 1) * samples_per_frame], y=frame, mode='lines',
                             name=f'Frame {i + 1}'))

fig.update_layout(title='Audio Frames', xaxis_title='Time (s)', yaxis_title='Amplitude')
fig.show()

def interpolate_frame_linear(frame, frame_length, num_points):
    time_original = np.arange(0, len(frame))
    time_interpolated = np.linspace(0, len(frame) - 1, num_points)
    interpolated_frame = np.interp(time_interpolated, time_original, frame)
    return interpolated_frame, time_interpolated

def interpolate_frame_cubic(frame, frame_length, num_points):
    time_original = np.arange(0, len(frame))
    time_interpolated = np.linspace(0, len(frame) - 1, num_points)
    cubic_spline = CubicSpline(time_original, frame)
    interpolated_frame = cubic_spline(time_interpolated)
    return interpolated_frame, time_interpolated

def interpolate_frame_quadratic(frame, frame_length, num_points):
    time_original = np.arange(0, len(frame))
    time_interpolated = np.linspace(0, len(frame) - 1, num_points)
    f = interp1d(time_original, frame, kind='quadratic')
    interpolated_frame = f(time_interpolated)
    return interpolated_frame, time_interpolated

def interpolate_frames_linear(frames, frame_length, sampling_rate, num_points):
    interpolated_frames = []
    for frame in frames:
        time_original = np.arange(0, len(frame))
        time_interpolated = np.linspace(0, len(frame) - 1, num_points)
        interpolated_frame = np.interp(time_interpolated, time_original, frame)
        interpolated_frames.append(interpolated_frame)
    interpolated_audio = np.concatenate(interpolated_frames)
    return interpolated_audio

def interpolate_frames_cubic(frames, frame_length, sampling_rate, num_points):
    interpolated_frames = []
    for frame in frames:
        interpolated_frame, _ = interpolate_frame_cubic(frame, frame_length, num_points)
        interpolated_frames.append(interpolated_frame)
    interpolated_audio = np.concatenate(interpolated_frames)
    return interpolated_audio

def interpolate_frames_quadratic(frames, frame_length, sampling_rate, num_points):
    interpolated_frames = []
    for frame in frames:
        interpolated_frame, _ = interpolate_frame_quadratic(frame, frame_length, num_points)
        interpolated_frames.append(interpolated_frame)
    interpolated_audio = np.concatenate(interpolated_frames)
    return interpolated_audio

frame_index = 100
chosen_frame = frames[frame_index]
num_points = 400

# Plot the frame and its interpolations
def plot_frame_and_interpolation(audio_data, sampling_rate, frame_index, num_points):
    frame_length = 0.02
    samples_per_frame = int(frame_length * sampling_rate)
    start_index = frame_index * samples_per_frame
    end_index = (frame_index + 1) * samples_per_frame

    # Original frame
    time = np.arange(start_index, end_index) / sampling_rate
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=audio_data[start_index:end_index], mode='lines', name='Original'))

    # Linearly interpolated frame
    frame = audio_data[start_index:end_index]
    interpolated_frame_linear, time_interpolated_linear = interpolate_frame_linear(frame, frame_length, num_points)
    time_interpolated_linear = time_interpolated_linear / sampling_rate + time[0]  # Adjust time scale
    fig.add_trace(
        go.Scatter(x=time_interpolated_linear, y=interpolated_frame_linear, mode='lines', name='Linear Interpolation'))

    # Cubic interpolated frame
    interpolated_frame_cubic, time_interpolated_cubic = interpolate_frame_cubic(frame, frame_length, num_points)
    time_interpolated_cubic = time_interpolated_cubic / sampling_rate + time[0]  # Adjust time scale
    fig.add_trace(
        go.Scatter(x=time_interpolated_cubic, y=interpolated_frame_cubic, mode='lines',
                   name='Cubic Spline Interpolation'))

    # Quadratic interpolated frame
    interpolated_frame_quadratic, time_interpolated_quadratic = interpolate_frame_quadratic(frame, frame_length,
                                                                                            num_points)
    time_interpolated_quadratic = time_interpolated_quadratic / sampling_rate + time[0]  # Adjust time scale
    fig.add_trace(
        go.Scatter(x=time_interpolated_quadratic, y=interpolated_frame_quadratic, mode='lines',
                   name='Quadratic Interpolation'))

    fig.update_layout(title=f'Frame {frame_index + 1} and Interpolations', xaxis_title='Time (s)',
                      yaxis_title='Amplitude')
    fig.show()

# Plot frame and its interpolations
plot_frame_and_interpolation(audio_data, sampling_rate, frame_index, num_points)

# Interpolate all frames
interpolated_audio_linear = interpolate_frames_linear(frames, frame_length, sampling_rate, num_points)
interpolated_audio_cubic = interpolate_frames_cubic(frames, frame_length, sampling_rate, num_points)
interpolated_audio_quadratic = interpolate_frames_quadratic(frames, frame_length, sampling_rate, num_points)

# Write the interpolated audio to new sound files
write("interpolated_audio_linear.wav", sampling_rate, np.int16(interpolated_audio_linear * 32767.0))
write("interpolated_audio_cubic.wav", sampling_rate, np.int16(interpolated_audio_cubic * 32767.0))
write("interpolated_audio_quadratic.wav", sampling_rate, np.int16(interpolated_audio_quadratic * 32767.0))
