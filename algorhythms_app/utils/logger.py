from torch.utils.tensorboard import SummaryWriter

class TrainingLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        
    def log_metrics(self, metrics, step):
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
            
    def log_audio(self, audio, step, sample_rate=44100):
        self.writer.add_audio('generated_audio', audio, step, sample_rate)
        
    def close(self):
        self.writer.close() 