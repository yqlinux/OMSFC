import pandas as pd
import os
import numpy as np

def split_mean_std(value):
    """将"数值±标准差"格式的字符串分解为数值和标准差"""
    try:
        if isinstance(value, str) and '±' in value:
            # 移除可能的星号标记
            value = value.replace('*', '')
            mean, std = value.split('±')
            return float(mean), float(std)
        return np.nan, np.nan
    except:
        return np.nan, np.nan

def calculate_row_averages(means_df):
    """计算每一行的Acc和ARI平均值"""
    acc_cols = [col for col in means_df.columns if 'Acc' in col]
    ari_cols = [col for col in means_df.columns if 'ARI' in col]
    
    row_acc_means = means_df[acc_cols].mean(axis=1)
    row_ari_means = means_df[ari_cols].mean(axis=1)
    
    return row_acc_means, row_ari_means

def process_excel_file(file_path):
    # 读取Excel文件中的所有工作表
    excel_file = pd.ExcelFile(file_path)
    sheet_names = excel_file.sheet_names
    
    # 存储所有工作表的数据
    all_means = []
    all_stds = []
    
    # 读取每个工作表的数据
    for sheet_name in sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # 创建存储均值和标准差的DataFrame
        means_df = pd.DataFrame(index=df.index, columns=df.columns)
        stds_df = pd.DataFrame(index=df.index, columns=df.columns)
        
        # 处理每个单元格
        for col in df.columns:
            for idx in df.index:
                value = str(df.loc[idx, col])
                mean, std = split_mean_std(value)
                means_df.loc[idx, col] = mean
                stds_df.loc[idx, col] = std
        
        all_means.append(means_df)
        all_stds.append(stds_df)
    
    # 确保所有工作表具有相同的结构
    if not all(df.shape == all_means[0].shape for df in all_means):
        raise ValueError("所有工作表必须具有相同的结构（行数和列数）")
    
    # 计算均值的平均值
    mean_sum = sum(df.fillna(0) for df in all_means)
    mean_count = sum((~df.isna()).astype(int) for df in all_means)
    final_means = mean_sum / mean_count
    
    # 计算合并后的标准差
    # 使用标准差传播公式：合并标准差 = sqrt(sum(std^2))/n
    std_squared_sum = sum(df.fillna(0)**2 for df in all_stds)
    final_stds = np.sqrt(std_squared_sum) / mean_count
    
    # 将结果组合为"均值±标准差"格式
    result_df = pd.DataFrame(index=final_means.index, columns=final_means.columns)
    for col in final_means.columns:
        for idx in final_means.index:
            mean = final_means.loc[idx, col]
            std = final_stds.loc[idx, col]
            if not (pd.isna(mean) or pd.isna(std)):
                result_df.loc[idx, col] = f"{mean:.2f}±{std:.2f}"
            else:
                result_df.loc[idx, col] = ""
    
    # 计算每一行的Acc和ARI平均值
    row_acc_means, row_ari_means = calculate_row_averages(final_means)
    
    # 创建包含行平均值的DataFrame
    row_averages_df = pd.DataFrame({
        'Row_Avg_Acc': row_acc_means,
        'Row_Avg_ARI': row_ari_means
    }, index=final_means.index)
    
    # 创建输出文件名
    base_name = os.path.splitext(file_path)[0]
    output_path = f"{base_name}_平均值.xlsx"
    row_avg_output_path = f"{base_name}_行平均值.xlsx"
    
    # 保存结果
    result_df.to_excel(output_path, index=True)
    row_averages_df.to_excel(row_avg_output_path, index=True)
    
    return output_path, row_avg_output_path

def main():
    file_path = "Comparison_Results_2025-03-14.xlsx"
    try:
        output_file, row_avg_file = process_excel_file(file_path)
        print(f"处理完成！")
        print(f"单元格平均值已保存至: {output_file}")
        print(f"行平均值已保存至: {row_avg_file}")
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")

if __name__ == "__main__":
    main() 