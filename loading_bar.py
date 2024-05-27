import sys
import time


class LoadingBar:
    def __init__(self, total: int, length: int = 40):
        self.total = total
        self.length = length
        self.progress = 0
        self.start_time = time.time()

    def update(self, progress: int, description: str = ''):
        self.progress = progress
        percent = self.progress / self.total
        filled_length = int(self.length * percent)

        bar = '=' * (filled_length - 1) + '>' + '-' * (self.length - filled_length)
        elapsed_time = time.time() - self.start_time
        remaining_time = (elapsed_time / percent) * (1 - percent) if percent > 0 else 0
        remaining_time_str = self.format_time(remaining_time)

        sys.stdout.write(f'\r{description} |{bar}| {percent * 100:.2f}% | ETA: {remaining_time_str}')
        sys.stdout.flush()

    def finish(self):
        self.update(self.total, "Complete")
        print()  # Move to the next line

    @staticmethod
    def format_time(seconds):
        mins, secs = divmod(seconds, 60)
        hrs, mins = divmod(mins, 60)
        if hrs > 0:
            return f"{int(hrs)}h {int(mins)}m {int(secs)}s"
        elif mins > 0:
            return f"{int(mins)}m {int(secs)}s"
        else:
            return f"{int(secs)}s"


if __name__ == '__main__':
    # Example usage:
    loading_bar = LoadingBar(total=100)

    for i in range(101):
        loading_bar.update(i, 'Processing')
        time.sleep(0.05)
    loading_bar.finish()