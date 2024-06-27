from qiskit import QuantumCircuit, Aer, execute
import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from scipy import signal
from tkinter import messagebox
import tensorflow as tf
import librosa
from sklearn.decomposition import PCA

# Wczytanie modelu klasyfikatora dźwięku
sound_classifier_model = tf.keras.models.load_model('sound_classifier_model.keras')

# Mapowanie częstotliwości na nazwy nut
frequency_to_note = {
    27.5: 'A0', 29.14: 'A#0/Bb0', 30.87: 'B0',
    32.703: 'C1', 34.648: 'C#1/Db1', 36.708: 'D1',
    38.891: 'D#1/Eb1', 41.203: 'E1', 43.654: 'F1',
    46.249: 'F#1/Gb1', 48.999: 'G1', 51.913: 'G#1/Ab1',
    55.0: 'A1', 58.27: 'A#1/Bb1', 61.735: 'B1',
    65.406: 'C2', 69.296: 'C#2/Db2', 73.416: 'D2',
    77.782: 'D#2/Eb2', 82.407: 'E2', 87.307: 'F2',
    92.499: 'F#2/Gb2', 97.999: 'G2', 103.826: 'G#2/Ab2',
    110.0: 'A2', 116.541: 'A#2/Bb2', 123.471: 'B2',
    130.813: 'C3', 138.591: 'C#3/Db3', 146.832: 'D3',
    155.563: 'D#3/Eb3', 164.814: 'E3', 174.614: 'F3',
    184.997: 'F#3/Gb3', 195.998: 'G3', 207.652: 'G#3/Ab3',
    220.0: 'A3', 233.082: 'A#3/Bb3', 246.942: 'B3',
    261.626: 'C4 (Middle C)', 277.183: 'C#4/Db4', 293.665: 'D4',
    311.127: 'D#4/Eb4', 329.628: 'E4', 349.228: 'F4',
    369.994: 'F#4/Gb4', 391.995: 'G4', 415.305: 'G#4/Ab4',
    440.0: 'A4', 466.164: 'A#4/Bb4', 493.883: 'B4',
    523.251: 'C5', 554.365: 'C#5/Db5', 587.33: 'D5',
    622.254: 'D#5/Eb5', 659.255: 'E5', 698.456: 'F5',
    739.989: 'F#5/Gb5', 783.991: 'G5', 830.609: 'G#5/Ab5',
    880.0: 'A5', 932.328: 'A#5/Bb5', 987.767: 'B5',
    1046.502: 'C6', 1108.731: 'C#6/Db6', 1174.659: 'D6',
    1244.508: 'D#6/Eb6', 1318.51: 'E6', 1396.913: 'F6',
    1479.978: 'F#6/Gb6', 1567.982: 'G6', 1661.219: 'G#6/Ab6',
    1760.0: 'A6', 1864.655: 'A#6/Bb6', 1975.533: 'B6',
    2093.005: 'C7', 2217.461: 'C#7/Db7', 2349.318: 'D7',
    2489.016: 'D#7/Eb7', 2637.02: 'E7', 2793.826: 'F7',
    2959.955: 'F#7/Gb7', 3135.963: 'G7', 3322.438: 'G#7/Ab7',
    3520.0: 'A7', 3729.31: 'A#7/Bb7', 3951.066: 'B7',
    4186.009: 'C8'
}

# Parametry układu kwantowego
num_qubits = 4
backend = Aer.get_backend('qasm_simulator')


def read_microphone_input(duration):
    # Funkcja do odczytu dźwięku z mikrofonu
    sample_rate = 44100
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return recording.flatten(), sample_rate


def extract_features(recording):
    # Funkcja do ekstrakcji cech dźwięku (MFCC)
    signal, sample_rate = recording, 44100  # Przykładowa próbka
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13)
    mfccs_normalized = librosa.util.normalize(mfccs)
    return np.mean(mfccs_normalized, axis=1)  # Uśrednienie cech MFCC



def generate_quantum_circuit(note_frequency):
    # Funkcja do generowania kwantowego obwodu
    qc = QuantumCircuit(num_qubits, num_qubits)
    # Implementacja obwodu kwantowego do pomiaru częstotliwości
    for qubit in range(num_qubits):
        qc.h(qubit)
        qc.u(note_frequency * 2 * np.pi / (2 ** qubit), 0, 0, qubit)
    qc.measure(range(num_qubits), range(num_qubits))
    qc.draw(output='mpl')
    plt.show()
    return qc


def generate_spectrogram(recording, sample_rate):
    # Funkcja do generowania spektrogramu
    frequencies, times, spectrogram = signal.spectrogram(recording, sample_rate)
    plt.pcolormesh(times, frequencies, np.log(spectrogram))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title('Spectrogram')
    plt.colorbar(label='Log intensity')
    plt.show()


def generate_frequency_plot(recording, sample_rate):
    # Funkcja do generowania wykresu częstotliwości
    frequencies, spectrum = signal.periodogram(recording, sample_rate)
    plt.plot(frequencies, np.log(spectrum))
    plt.ylabel('Log Intensity')
    plt.xlabel('Frequency [Hz]')
    plt.title('Frequency Spectrum')
    plt.show()


def process_sound_quantum(label_text, recording, sample_rate, duration):
    # Funkcja do przetwarzania dźwięku kwantowo
    if len(recording) == 0:
        messagebox.showerror("Error", "No sound recorded. Please try again.")
        return

    generate_spectrogram(recording, sample_rate)
    generate_frequency_plot(recording, sample_rate)

    frequency = np.fft.fft(recording)
    dominant_frequency_index = np.argmax(np.abs(frequency))
    dominant_frequency = dominant_frequency_index / duration

    if dominant_frequency == 0:
        messagebox.showerror("Error", "Unable to detect dominant frequency. Please try again.")
        return

    closest_frequency = None
    for f in frequency_to_note:
        if f >= dominant_frequency:
            closest_frequency = f
            break

    if closest_frequency is None:
        messagebox.showerror("Error", "No matching frequency found for the note.")
        return

    closest_note = frequency_to_note.get(closest_frequency, "Unknown")

    print("Dominant Frequency:", dominant_frequency, "Hz")
    print("Closest Frequency:", closest_frequency, "Hz")
    label_text.set("Closest Note: {}".format(closest_note))

    qc = generate_quantum_circuit(closest_frequency)
    job = execute(qc, backend, shots=1024)
    result = job.result()
    counts = result.get_counts(qc)
    print("Measurement results:", counts)


def classify_sound_from_file(file_path):
    # Wczytaj plik dźwiękowy
    signal, sample_rate = librosa.load(file_path, sr=None)

    # Przetwórz cechy dźwięku
    features = extract_features(signal)

    predicted_label = sound_classifier_model.predict(features.reshape(1, -1))

    # Wyświetlenie etykiety liczbowej w konsoli
    print("Predicted Label (numeric):", predicted_label[0])

    return predicted_label[0]


class MainMenuApp:
    def __init__(self, master):
        self.master = master
        master.title("Main Menu")

        self.label = tk.Label(master, text="Choose an option:")
        self.label.pack()

        self.piano_button = tk.Button(master, text="Piano Note Recognizer", command=self.start_piano_recognizer)
        self.piano_button.pack()

        self.catdog_button = tk.Button(master, text="Cat/Dog Recognizer", command=self.start_catdog_recognizer)
        self.catdog_button.pack()

        self.quit_button = tk.Button(master, text="Quit", command=self.quit_program)
        self.quit_button.pack()

    def start_piano_recognizer(self):
        self.master.withdraw()
        root_piano = tk.Toplevel(self.master)
        app_piano = PianoNoteRecognizerApp(root_piano)

    def start_catdog_recognizer(self):
        self.master.withdraw()
        root_catdog = tk.Toplevel(self.master)
        app_catdog = CatDogRecognizerApp(root_catdog)

    def quit_program(self):
        self.master.quit()

class CatDogRecognizerApp:
    def __init__(self, master):
        self.master = master
        master.title("Cat/Dog Recognizer")

        self.label = tk.Label(master, text="Click 'Browse' to select an audio file.")
        self.label.pack()

        self.browse_button = tk.Button(master, text="Browse", command=self.browse_audio_file)
        self.browse_button.pack()

        self.result_label = tk.Label(master, text="")
        self.result_label.pack()

        self.back_button = tk.Button(master, text="Back to Menu", command=self.back_to_menu)
        self.back_button.pack()

    def browse_audio_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav;*.mp3;*.flac")])
        if file_path:
            predicted_label = classify_sound_from_file(file_path)
            if predicted_label < 0.49:
                self.result_label.config(text="This sound is classified as: Cat")
            else:
                self.result_label.config(text="This sound is classified as: Dog")

    def back_to_menu(self):
        self.master.destroy()
        root_menu = tk.Tk()
        app_menu = MainMenuApp(root_menu)



class PianoNoteRecognizerApp:
    def __init__(self, master):
        self.master = master
        master.title("Piano Note Recognizer")

        self.label = tk.Label(master, text="Click 'Recognize Note' to start recognizing the piano note.")
        self.label.pack()

        self.note_label_text = tk.StringVar()
        self.note_label = tk.Label(master, textvariable=self.note_label_text)
        self.note_label.pack()

        self.recognize_button = tk.Button(master, text="Recognize Note", command=self.recognize_note)
        self.recognize_button.pack()

        self.back_button = tk.Button(master, text="Back to Menu", command=self.back_to_menu)
        self.back_button.pack()

    def recognize_note(self):
        self.note_label_text.set("")
        recording, sample_rate = read_microphone_input(2)  # Ustawienie czasu trwania nagrania
        process_sound_quantum(self.note_label_text, recording, sample_rate,
                              2)  # Przekazanie nagrania, częstotliwości próbkowania i czasu trwania

    def back_to_menu(self):
        # Zamyka okno aplikacji pianina i otwiera główne menu
        self.master.destroy()
        root_menu = tk.Tk()
        app_menu = MainMenuApp(root_menu)


def main():
    root = tk.Tk()
    app = MainMenuApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()