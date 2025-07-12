from teacher_network import load_teacher_model
from student_network import StudentNet
from transformer import get_transform
from dataset import ImageDataset
from utils import compute_ssim

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher = load_teacher_model().to(device)
    student = StudentNet().to(device)
    transform = get_transform()
    
    train_dataset = ImageDataset(transform=transform, mode='train')
    test_dataset = ImageDataset(transform=transform, mode='test')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    optimizer = optim.Adam(student.parameters(), lr=1e-3)
    
    for epoch in range(10):
        student.train()
        for img in train_loader:
            img = img.to(device)
            with torch.no_grad():
                teacher_output = teacher(img)
            student_output = student(img)
            loss = F.mse_loss(student_output, teacher_output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} done.")
    
    # Evaluation
    student.eval()
    ssim_scores = []
    with torch.no_grad():
        for img in test_loader:
            img = img.to(device)
            teacher_output = teacher(img)
            student_output = student(img)
            ssim = compute_ssim(student_output, teacher_output)
            ssim_scores.append(ssim.item())
    print(f"Mean SSIM: {sum(ssim_scores)/len(ssim_scores):.4f}")

if __name__ == "__main__":
    train()