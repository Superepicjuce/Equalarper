
import threading
import numpy as np
import pygame
import sounddevice as sd

# ---------------- CONFIGURATION ----------------
WIDTH, HEIGHT = 1280, 720
BUFFER_SIZE = 8192
FPS = 60

BG = (2, 10, 6)
GREEN = (100, 255, 145)
DIM_GREEN = (20, 75, 42)
BRIGHT_GREEN = (190, 255, 205)

# ---------------- AUDIO INPUT ----------------
device = sd.query_devices(kind="input")
channels = min(2, int(device["max_input_channels"]))

if channels < 1:
    raise RuntimeError("No audio input device is available.")

audio = np.zeros((BUFFER_SIZE, channels), dtype=np.float32)
write_pos = 0
audio_lock = threading.Lock()


def audio_callback(indata, frames, time_info, status):
    global write_pos

    n = len(indata)
    with audio_lock:
        if n >= BUFFER_SIZE:
            audio[:] = indata[-BUFFER_SIZE:]
            write_pos = 0
        else:
            end = write_pos + n

            if end <= BUFFER_SIZE:
                audio[write_pos:end] = indata
            else:
                first = BUFFER_SIZE - write_pos
                audio[write_pos:] = indata[:first]
                audio[:n - first] = indata[first:]

            write_pos = end % BUFFER_SIZE


def get_audio():
    with audio_lock:
        return np.concatenate(
            (audio[write_pos:], audio[:write_pos]), axis=0
        ).copy()


# ---------------- GRAPHICS ----------------
pygame.init()
pygame.display.set_caption("Equalarper // LARP GRAPH")

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT), pygame.RESIZABLE
)
clock = pygame.time.Clock()
font = pygame.font.SysFont("monospace", 17)
small_font = pygame.font.SysFont("monospace", 13)

trail = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
gain = 1.0
mode = "WAVE"
fullscreen = False


def draw_grid(surface, w, h):
    cx, cy = w // 2, h // 2

    for i in range(1, 10):
        x = int(w * i / 10)
        pygame.draw.line(surface, DIM_GREEN, (x, 55), (x, h - 45))

    for i in range(1, 9):
        y = int(h * i / 9)
        pygame.draw.line(surface, DIM_GREEN, (25, y), (w - 25, y))

    pygame.draw.line(surface, GREEN, (25, cy), (w - 25, cy), 1)
    pygame.draw.line(surface, GREEN, (cx, 55), (cx, h - 45), 1)

    # Center marker
    pygame.draw.circle(surface, BRIGHT_GREEN, (cx, cy), 3)


def draw_glow(surface, points, color=GREEN):
    if len(points) < 2:
        return

    # Broad, faint glow followed by a crisp phosphor trace.
    pygame.draw.lines(surface, (12, 65, 30), False, points, 9)
    pygame.draw.lines(surface, (30, 150, 70), False, points, 4)
    pygame.draw.lines(surface, color, False, points, 2)


def main():
    global gain, mode, fullscreen, screen, trail

    with sd.InputStream(
        device=None,
        channels=channels,
        samplerate=44100,
        blocksize=512,
        dtype="float32",
        callback=audio_callback,
    ):
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_w:
                        mode = "WAVE"
                    elif event.key == pygame.K_x:
                        mode = "XY"
                    elif event.key == pygame.K_UP:
                        gain = min(8.0, gain * 1.2)
                    elif event.key == pygame.K_DOWN:
                        gain = max(0.1, gain / 1.2)
                    elif event.key == pygame.K_f:
                        fullscreen = not fullscreen
                        if fullscreen:
                            screen = pygame.display.set_mode(
                                (0, 0), pygame.FULLSCREEN
                            )
                        else:
                            screen = pygame.display.set_mode(
                                (WIDTH, HEIGHT), pygame.RESIZABLE
                            )
                        trail = pygame.Surface(
                            screen.get_size(), pygame.SRCALPHA
                        )

            w, h = screen.get_size()

            # Phosphor persistence: gently fade the previous frame.
            trail.fill((0, 0, 0, 35))
            screen.blit(trail, (0, 0))
            draw_grid(screen, w, h)

            samples = get_audio()
            left = samples[:, 0]

            if channels >= 2:
                right = samples[:, 1]
            else:
                right = left

            left = np.clip(left * gain, -1.0, 1.0)
            right = np.clip(right * gain, -1.0, 1.0)

            if mode == "WAVE":
                # Show the most recent portion of the waveform.
                count = min(len(left), w - 80)
                indices = np.linspace(
                    len(left) - count, len(left) - 1,
                    count, dtype=int
                )

                x_positions = np.linspace(
                    40, w - 40, count, dtype=int
                )
                amplitude = h * 0.34

                points = [
                    (
                        int(x),
                        int(h / 2 - left[i] * amplitude)
                    )
                    for x, i in zip(x_positions, indices)
                ]

            else:
                # XY mode: left channel vs. right channel.
                count = min(len(left), 3000)
                indices = np.linspace(
                    len(left) - count, len(left) - 1,
                    count, dtype=int
                )
                scale = min(w, h) * 0.38

                points = [
                    (
                        int(w / 2 + left[i] * scale),
                        int(h / 2 - right[i] * scale)
                    )
                    for i in indices
                ]

            draw_glow(screen, points)

            # Header and readout
            title = font.render(
                "Equalarper // LARP GRAPH",
                True, BRIGHT_GREEN
            )
            screen.blit(title, (30, 18))

            readout = small_font.render(
                f"MODE: {mode}   GAIN: {gain:.1f}x"
                f"   INPUT: {channels} CH",
                True, GREEN
            )
            screen.blit(readout, (30, h - 35))

            controls = small_font.render(
                "W: WAVE   X: XY   UP/DOWN: GAIN"
                "   F: FULLSCREEN   ESC: QUIT",
                True, GREEN
            )
            screen.blit(
                controls,
                (max(30, w - controls.get_width() - 30), h - 35)
            )

            pygame.display.flip()
            clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pygame.quit()
