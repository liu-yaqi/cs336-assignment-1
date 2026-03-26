import numpy as np
import numpy.typing as npt
import torch
from pathlib import Path


class Dataset:
    """
    数据集类，用于加载和管理大模型训练数据
    """
    
    def __init__(
        self,
        dataset: npt.NDArray
    ):
        """
        初始化Dataset实例
        
        Args:
            dataset (npt.NDArray): 1D numpy数组，包含token IDs
        """
        self.dataset = dataset
        self.data_length = len(self.dataset)
    
    def __len__(self) -> int:
        """
        返回数据集的token总数
        
        Returns:
            int: 数据集长度
        """
        return len(self.dataset)
    
    def get_batch(
        self,
        batch_size: int,
        context_length: int,
        device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        从数据集中随机采样一个batch
        
        给定batch_size和context_length，从数据集中随机采样语言模型输入序列
        及其对应的标签。
        
        Args:
            batch_size (int): batch大小
            context_length (int): 上下文长度
            device (str): PyTorch设备字符串 (e.g., 'cpu' or 'cuda:0')
        
        Returns:
            tuple: (inputs, labels)，两个形状均为 (batch_size, context_length) 的torch.LongTensor
                  - inputs: 输入序列
                  - labels: 对应的标签（输入往后移动一位）
        """
        inputs = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)
        labels = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)

        for i in range(batch_size):
            # 随机采样起始位置，确保有足够的token用于context_length + 1（label）
            start_idx = np.random.randint(0, self.data_length - context_length) # 最大是self.data_length - context_length -1
            end_idx = start_idx + context_length
            
            # input: dataset[start_idx:end_idx]
            # label: dataset[start_idx+1:end_idx+1] (往后移动一位作为预测目标)
            inputs[i] = torch.tensor(self.dataset[start_idx:end_idx], dtype=torch.long, device=device)
            labels[i] = torch.tensor(self.dataset[start_idx + 1:end_idx + 1], dtype=torch.long, device=device)

        return inputs, labels
    
    def create_iterator(
        self,
        batch_size: int,
        context_length: int,
        device: str,
        num_batches: int | None = None
    ):
        """
        创建数据迭代器
        
        Args:
            batch_size (int): batch大小
            context_length (int): 上下文长度
            device (str): PyTorch设备
            num_batches (int | None): 生成的batch数量，None表示持续迭代
        
        Yields:
            tuple: (inputs, labels) batch数据
        """
        batch_count = 0
        while num_batches is None or batch_count < num_batches:
            inputs, labels = self.get_batch(batch_size, context_length, device)
            yield inputs, labels
            batch_count += 1