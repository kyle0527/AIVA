#!/usr/bin/env python3
"""分析三個權重檔案的內容結構"""
import torch
from pathlib import Path

weight_files = [
    "models/weights/aiva_real_ai_core.pth",
    "models/weights/aiva_real_weights.pth",
    "weights/models/aiva_scalable_bionet_latest.pth"
]

for weight_file in weight_files:
    filepath = Path(weight_file)
    if not filepath.exists():
        print(f"\n❌ 檔案不存在: {weight_file}")
        continue
    
    print(f"\n{'='*80}")
    print(f"📁 檔案: {weight_file}")
    print(f"💾 大小: {filepath.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"{'='*80}")
    
    try:
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # 檢查是否為字典格式
        if isinstance(checkpoint, dict):
            print(f"✅ 格式: Dictionary (keys: {list(checkpoint.keys())})")
            
            # 檢查是否有 model_state_dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"📦 結構: 標準 checkpoint 格式")
                if 'total_params' in checkpoint:
                    print(f"🔢 記錄參數數: {checkpoint['total_params']:,}")
                if 'epoch' in checkpoint:
                    print(f"🔄 訓練輪數: {checkpoint['epoch']}")
            else:
                state_dict = checkpoint
                print(f"📦 結構: 直接 state_dict 格式")
        else:
            state_dict = checkpoint
            print(f"⚠️  格式: 非字典類型 ({type(checkpoint)})")
        
        # 分析 state_dict
        print(f"\n🧠 模型層結構:")
        total_params = 0
        layer_info = {}
        
        for key, tensor in state_dict.items():
            print(f"  {key:50s} {str(tuple(tensor.shape)):25s} {tensor.numel():>12,} params")
            total_params += tensor.numel()
            
            # 提取層名稱
            layer_name = key.split('.')[0]
            if layer_name not in layer_info:
                layer_info[layer_name] = 0
            layer_info[layer_name] += tensor.numel()
        
        print(f"\n📊 統計:")
        print(f"  總層數: {len(state_dict)}")
        print(f"  總參數: {total_params:,} ({total_params/1_000_000:.2f}M)")
        print(f"  預期大小: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
        
        print(f"\n🏗️  主要組件:")
        for layer, params in sorted(layer_info.items(), key=lambda x: x[1], reverse=True):
            print(f"  {layer:30s} {params:>12,} params ({params/total_params*100:>5.1f}%)")
        
        # 判斷模型類型
        print(f"\n🔍 模型類型推斷:")
        if 'layer1.weight' in state_dict:
            input_dim = state_dict['layer1.weight'].shape[1]
            output_dim = state_dict['layer1.weight'].shape[0]
            print(f"  ➡️  第一層: {input_dim} -> {output_dim}")
        
        if 'output_layer.weight' in state_dict:
            final_output = state_dict['output_layer.weight'].shape[0]
            print(f"  ⬅️  輸出層: -> {final_output}")
        elif 'aux_output.weight' in state_dict:
            aux_output = state_dict['aux_output.weight'].shape[0]
            print(f"  ⬅️  輔助輸出: -> {aux_output}")
        
        # 檢查特殊層
        has_bn = any('bn' in key for key in state_dict.keys())
        has_dropout = any('dropout' in key for key in state_dict.keys())
        has_aux = any('aux' in key for key in state_dict.keys())
        
        print(f"\n✨ 特殊層:")
        print(f"  BatchNorm: {'✅' if has_bn else '❌'}")
        print(f"  Dropout: {'✅' if has_dropout else '❌'}")
        print(f"  輔助輸出: {'✅' if has_aux else '❌'}")
        
    except Exception as e:
        print(f"❌ 載入失敗: {e}")

print(f"\n{'='*80}")
print("分析完成")
print(f"{'='*80}")
