import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CustomDataset
from model import SRCNN
import cv2
import matplotlib.pyplot as plt
import numpy as np  # NumPy 임포트 추가

# 파라미터 설정
n1, n2, n3 = 128, 64, 3
f1, f2, f3 = 9, 3, 5
upscale_factor = 3

input_size = 33
output_size = input_size - f1 - f2 - f3 + 3
stride = 14

batch_size = 128
epochs = 200
path = r"C:\\Users\\User\\Desktop\\pyth\\T91"  # 데이터셋 경로
save_path = r"C:\Users\User\Desktop\pyth\checkpoint_epoch_90_batch_1 - 복사본.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 학습 함수 정의
def train(dataloader, model, loss_fn, optimizer, epoch, min_loss):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch + 1}/{len(dataloader)}] - loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

            if loss < min_loss:
                min_loss = loss
                torch.save(model.state_dict(), f"checkpoint_epoch_{epoch + 1}_batch_{batch + 1}.pth")
                print(f"Checkpoint saved at epoch {epoch + 1}, batch {batch + 1} with loss: {loss:>7f}")

    return min_loss

# 모델 학습 및 테스트
if __name__ == '__main__':
    # 데이터셋 생성 및 DataLoader 정의
    # dataset = CustomDataset(img_paths=path, input_size=input_size, output_size=output_size, stride=stride, upscale_factor=upscale_factor)
    # train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # model = SRCNN(kernel_list=[f1, f2, f3], filters_list=[n1, n2, n3]).to(device)
    # print(model)

    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # loss_fn = torch.nn.MSELoss()

    # min_loss = float('inf')  # 무한대로 초기화

    # for epoch in range(epochs):
    #     print(f"{epoch + 1} Epochs ...")
    #     model.train()
    #     min_loss = train(train_dataloader, model, loss_fn, optimizer, epoch, min_loss)

    # print("Done!")

    # torch.save(model.state_dict(), save_path)

    # 결과 시각화 및 이미지 출력 코드
    hr_img_path = r"C:\Users\User\Desktop\pyth\Set14\Set14\monarch.png"
    hr_img = cv2.imread(hr_img_path)
    hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)

    hr_img = hr_img.astype(np.float32) / 255.0
    temp_img = cv2.resize(hr_img, dsize=(0, 0), fx=1/upscale_factor, fy=1/upscale_factor, interpolation=cv2.INTER_AREA)
    bicubic_img = cv2.resize(temp_img, dsize=(0, 0), fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC)

    model_t = SRCNN([f1, f2, f3], [n1, n2, n3]).to(device)
    model_t.load_state_dict(torch.load(save_path))
    model_t.eval()

    input_img = bicubic_img.transpose((2, 0, 1))
    input_img = torch.tensor(input_img).unsqueeze(0).to(device)

    with torch.no_grad():
        srcnn_img = model_t(input_img).cpu().numpy()

    srcnn_img = srcnn_img.squeeze(0).transpose((1, 2, 0))
    srcnn_img = np.clip(srcnn_img, 0, 1)
    srcnn_img = (srcnn_img * 255).astype(np.uint8)

    # 결과 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(hr_img)
    axes[0].set_title('HR Image')
    axes[0].axis('off')

    axes[1].imshow(bicubic_img)
    axes[1].set_title('Bicubic Image')
    axes[1].axis('off')

    axes[2].imshow(srcnn_img)
    axes[2].set_title('SRCNN Image')
    axes[2].axis('off')

    plt.show()
