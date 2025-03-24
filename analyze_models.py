import pandas as pd
import numpy as np
import torch
from chronos import BaseChronosPipeline
import math
import os
import sys
import json
from datetime import datetime
import logging
from tqdm import tqdm

# 设置日志系统
def setup_logging():
    """配置日志系统"""
    logger = logging.getLogger('chronos')
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器，记录到文件
    if not os.path.exists('logs'):
        os.makedirs('logs')
    log_file = os.path.join('logs', f'chronos_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # 创建控制台处理器，只显示错误信息
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.CRITICAL)  # 提高控制台日志级别，只显示严重错误，保持终端干净只显示进度条
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 将处理器添加到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    print(f"日志记录到: {log_file}")
    return logger

# 全局日志对象
logger = setup_logging()

def setup_date_ranges(input_start='2023-03', input_end='2023-12', 
                      predict_start='2024-01', predict_end='2024-12'):
    """
    设置输入数据范围和预测范围
    
    参数:
    input_start : str, 输入数据的开始日期 (格式: 'YYYY-MM')
    input_end : str, 输入数据的结束日期 (格式: 'YYYY-MM')
    predict_start : str, 预测的开始日期 (格式: 'YYYY-MM')
    predict_end : str, 预测的结束日期 (格式: 'YYYY-MM')
    
    返回:
    dict: 包含所有日期范围信息的字典
    """
    # 解析日期
    input_start_date = pd.to_datetime(input_start)
    input_end_date = pd.to_datetime(input_end)
    predict_start_date = pd.to_datetime(predict_start)
    predict_end_date = pd.to_datetime(predict_end)
    
    # 生成输入日期范围
    input_months = pd.date_range(start=input_start_date, end=input_end_date, freq='ME')
    
    # 生成预测月份列表
    predict_months = []
    current = predict_start_date
    while current <= predict_end_date:
        predict_months.append(current.strftime('%b').upper())
        current = current + pd.DateOffset(months=1)
    
    logger.info(f"日期范围配置: 输入 {input_start} 到 {input_end}, 预测 {predict_start} 到 {predict_end}")
    
    return {
        'input_start': input_start,
        'input_end': input_end,
        'predict_start': predict_start,
        'predict_end': predict_end,
        'input_months': input_months,
        'predict_year': predict_start_date.year,
        'predict_months': predict_months
    }

def filter_device_data(df, device_type, date_ranges=None):
    """
    过滤并处理单个设备类型的数据
    
    参数:
    df : DataFrame, 原始数据
    device_type : str, 设备类型
    date_ranges : dict, 日期范围信息，如果为None则使用默认值
    """
    logger.info(f"正在过滤 {device_type} 的数据...")
    
    # 如果没有提供日期范围，使用默认值
    if date_ranges is None:
        date_ranges = setup_date_ranges()
    
    try:
        # 过滤设备数据
        filtered_df = df[df["计划机型"] == device_type]
        
        if len(filtered_df) == 0:
            logger.warning(f"找不到设备 {device_type} 的数据")
            return None
            
        logger.debug(f"找到 {device_type} 数据，行数: {len(filtered_df)}")
        
        # 检查是否所有值都是0
        data_columns = [col for col in filtered_df.columns if col not in ['计划机型', '关键指标']]
        if (filtered_df[data_columns] == 0).all().all():
            logger.warning(f"计划机型 {device_type} 所有数据都是0，跳过处理。")
            return None
    
        # 创建结果DataFrame
        result_data = []
        
        # 处理输入数据范围内的数据
        input_count = 0
        for date in date_ranges['input_months']:
            month_col = date.strftime('%b').upper() + ' ' + date.strftime('%Y')
            if month_col in filtered_df.columns:
                try:
                    value = filtered_df[month_col].iloc[2]
                    row = {
                        'Month': date.strftime('%Y-%m'),
                        'Actual': int(value) if pd.notna(value) else 0,
                        'IBP': None
                    }
                    result_data.append(row)
                    input_count += 1
                except Exception as e:
                    logger.warning(f"处理 {month_col} 时出错: {e}")
        
        logger.debug(f"处理了 {input_count} 个输入月份数据")
        
        # 处理预测范围内的数据
        predict_count = 0
        predict_year = date_ranges['predict_year']
        for month in date_ranges['predict_months']:
            month_col = f'{month} {predict_year}'
            if month_col in filtered_df.columns:
                try:
                    actual_value = filtered_df[month_col].iloc[2]
                    ibp_value = filtered_df[month_col].iloc[0]
                    row = {
                        'Month': f'{predict_year}-{pd.to_datetime(f"{predict_year}-{month}-01").strftime("%m")}',
                        'Actual': int(actual_value) if pd.notna(actual_value) else 0,
                        'IBP': float(ibp_value) if pd.notna(ibp_value) else 0
                    }
                    result_data.append(row)
                    predict_count += 1
                except Exception as e:
                    logger.warning(f"处理 {month_col} 时出错: {e}")
        
        logger.debug(f"处理了 {predict_count} 个预测月份数据")
        
        if len(result_data) == 0:
            logger.warning(f"设备 {device_type} 没有有效数据")
            return None
            
        result_df = pd.DataFrame(result_data)
        logger.info(f"设备 {device_type} 数据处理完成，共 {len(result_df)} 行")
        return result_df
        
    except Exception as e:
        logger.error(f"处理设备 {device_type} 数据时发生异常: {e}")
        return None

def analyze_device(df, pipeline, device_type, date_ranges=None):
    """
    分析单个设备的数据并返回预测结果
    
    参数:
    df : DataFrame, 处理后的设备数据
    pipeline : Chronos模型
    device_type : str, 设备类型
    date_ranges : dict, 日期范围信息，如果为None则使用默认值
    """
    # 如果没有提供日期范围，使用默认值
    if date_ranges is None:
        date_ranges = setup_date_ranges()
        
    # 设置时间索引
    df['Month'] = pd.to_datetime(df['Month'])
    df = df.set_index('Month')
    
    # 设置回测参数
    test_start = date_ranges['predict_start']
    test_end = date_ranges['predict_end']
    forecast_horizon = 1
    step_size = 1
    
    logger.info(f"{device_type} 开始滚动预测: {test_start} 到 {test_end}")
    
    # 准备存储预测结果
    all_predictions = []
    all_actuals = []
    all_dates = []
    all_lower_bounds = []
    all_upper_bounds = []
    
    # 滚动预测
    current_date = pd.to_datetime(test_start)
    end_date = pd.to_datetime(test_end)
    
    try:
        # 计算总步数以设置进度条
        total_steps = 0
        temp_date = current_date
        while temp_date <= end_date:
            total_steps += 1
            temp_date = temp_date + pd.DateOffset(months=step_size)
        
        # 创建预测进度条
        predict_bar = tqdm(total=total_steps, desc=f"预测 {device_type}", leave=False, ncols=80)
        
        step_count = 0
        while current_date <= end_date:
            train_data = df[:current_date]
            step_count += 1
            
            if current_date not in df.index:
                logger.warning(f"{current_date.strftime('%Y-%m-%d')} 不在数据索引中，提前结束预测")
                break
            
            # 更新进度条描述
            predict_bar.set_description(f"预测 {device_type}: {current_date.strftime('%Y-%m')}")
                
            # 检查训练数据是否有效
            if len(train_data["Actual"]) == 0:
                logger.warning(f"训练数据为空，跳过 {current_date.strftime('%Y-%m-%d')}")
                current_date = current_date + pd.DateOffset(months=step_size)
                predict_bar.update(1)
                continue
                
            # 模型预测
            try:
                quantiles, mean = pipeline.predict_quantiles(
                    context=torch.tensor(train_data["Actual"].values),
                    prediction_length=forecast_horizon,
                    quantile_levels=[0.1, 0.5, 0.9],
                )
                
                low = quantiles[0, :forecast_horizon, 0].cpu().numpy()
                median = quantiles[0, :forecast_horizon, 1].cpu().numpy()
                high = quantiles[0, :forecast_horizon, 2].cpu().numpy()
                
                prediction_date = current_date
                all_predictions.append(median[0])
                all_lower_bounds.append(low[0])
                all_upper_bounds.append(high[0])
                all_dates.append(prediction_date)
                
                if prediction_date in df.index:
                    all_actuals.append(df.loc[prediction_date, 'Actual'])
                else:
                    all_actuals.append(np.nan)
                
            except Exception as e:
                logger.warning(f"预测错误: {e}")
                all_predictions.append(np.nan)
                all_lower_bounds.append(np.nan)
                all_upper_bounds.append(np.nan)
                all_dates.append(current_date)
                
                if current_date in df.index:
                    all_actuals.append(df.loc[current_date, 'Actual'])
                else:
                    all_actuals.append(np.nan)
            
            current_date = current_date + pd.DateOffset(months=step_size)
            predict_bar.update(1)
        
        # 关闭预测进度条
        predict_bar.close()
        
        logger.info(f"{device_type} 预测完成，共 {len(all_predictions)} 个预测点")
        
        # 获取IBP值
        ibp_values = np.array([])
        try:
            ibp_values = df[test_start:test_end]['IBP'].values
        except Exception as e:
            logger.warning(f"获取IBP值出错: {e}")
            ibp_values = np.array([np.nan] * len(all_predictions))
        
        # 确保数组长度一致
        n = min(len(all_predictions), len(all_actuals), len(all_dates), 
                len(all_lower_bounds), len(all_upper_bounds), len(ibp_values))
        
        return {
            'dates': pd.DatetimeIndex(all_dates[:n]),
            'predictions': np.array(all_predictions[:n]),
            'actuals': np.array(all_actuals[:n]),
            'lower_bounds': np.array(all_lower_bounds[:n]),
            'upper_bounds': np.array(all_upper_bounds[:n]),
            'ibp_values': ibp_values[:n] if len(ibp_values) > 0 else np.array([np.nan] * n),
            'historical_dates': df.index,
            'historical_values': df["Actual"]
        }
        
    except Exception as e:
        logger.error(f"{device_type} 预测过程中发生错误: {e}")
        # 返回空结果
        return {
            'dates': pd.DatetimeIndex([]),
            'predictions': np.array([]),
            'actuals': np.array([]),
            'lower_bounds': np.array([]),
            'upper_bounds': np.array([]),
            'ibp_values': np.array([]),
            'historical_dates': df.index,
            'historical_values': df["Actual"]
        }

def save_results_to_json(results_dict, device_types, output_dir, input_start, input_end, predict_start, predict_end):
    """
    将预测结果保存为JSON文件
    
    参数:
    results_dict : dict, 包含每个设备预测结果的字典 {device_type: results}
    device_types : list, 设备类型列表
    output_dir : str, 输出目录
    input_start : str, 输入数据的开始日期
    input_end : str, 输入数据的结束日期
    predict_start : str, 预测的开始日期
    predict_end : str, 预测的结束日期
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建可序列化的结果字典
    serializable_results = {}
    
    # 用于统计摘要信息
    overall_metrics = {
        'processed_devices': 0,
        'chronos_wins': 0,
        'ibp_wins': 0
    }
    
    # 处理每个设备的结果
    for device_type in device_types:
        if device_type in results_dict:
            result = results_dict[device_type]
            overall_metrics['processed_devices'] += 1
            
            # 将NumPy数组和Pandas日期序列转换为可序列化的格式
            dates = [d.strftime('%Y-%m-%d') for d in result['dates']]
            historical_dates = [d.strftime('%Y-%m-%d') for d in result['historical_dates']]
            
            # 计算评估指标
            valid_indices = ~np.isnan(result['actuals'])
            if np.any(valid_indices):
                actuals = result['actuals'][valid_indices].tolist()
                predictions = result['predictions'][valid_indices].tolist()
                
                # 计算Chronos的MAE和RMSE
                chronos_mae = np.mean(np.abs(result['actuals'][valid_indices] - result['predictions'][valid_indices]))
                chronos_rmse = np.sqrt(np.mean((result['actuals'][valid_indices] - result['predictions'][valid_indices])**2))
                
                # 计算Chronos的MAPE
                valid_mape_indices = valid_indices & (result['actuals'] != 0)
                if np.any(valid_mape_indices):
                    chronos_mape = np.mean(np.abs((result['actuals'][valid_mape_indices] - 
                                               result['predictions'][valid_mape_indices]) / 
                                               result['actuals'][valid_mape_indices])) * 100
                else:
                    chronos_mape = None
                
                # 计算IBP的评估指标
                valid_ibp_indices = ~np.isnan(result['ibp_values'])
                if np.any(valid_ibp_indices):
                    ibp_mae = np.mean(np.abs(result['actuals'][valid_ibp_indices] - result['ibp_values'][valid_ibp_indices]))
                    ibp_rmse = np.sqrt(np.mean((result['actuals'][valid_ibp_indices] - result['ibp_values'][valid_ibp_indices])**2))
                    
                    # 计算IBP的MAPE
                    valid_ibp_mape_indices = valid_ibp_indices & (result['actuals'] != 0)
                    if np.any(valid_ibp_mape_indices):
                        ibp_mape = np.mean(np.abs((result['actuals'][valid_ibp_mape_indices] - 
                                                 result['ibp_values'][valid_ibp_mape_indices]) / 
                                                 result['actuals'][valid_ibp_mape_indices])) * 100
                    else:
                        ibp_mape = None
                else:
                    ibp_mae = None
                    ibp_rmse = None
                    ibp_mape = None
                
                # 判断胜出方
                winner = "N/A"
                if chronos_mae is not None and ibp_mae is not None:
                    chronos_wins = 0
                    ibp_wins = 0
                    
                    if chronos_mae < ibp_mae:
                        chronos_wins += 1
                    elif ibp_mae < chronos_mae:
                        ibp_wins += 1
                        
                    if chronos_rmse < ibp_rmse:
                        chronos_wins += 1
                    elif ibp_rmse < chronos_rmse:
                        ibp_wins += 1
                        
                    if chronos_mape is not None and ibp_mape is not None:
                        if chronos_mape < ibp_mape:
                            chronos_wins += 1
                        elif ibp_mape < chronos_mape:
                            ibp_wins += 1
                    
                    if chronos_wins >= 2:
                        winner = "Chronos"
                        overall_metrics['chronos_wins'] += 1
                    elif ibp_wins >= 2:
                        winner = "IBP"
                        overall_metrics['ibp_wins'] += 1
                
                # 保存评估指标和预测值
                serializable_results[device_type] = {
                    'dates': dates,
                    'actual_values': result['actuals'].tolist(),
                    'chronos_predictions': result['predictions'].tolist(),
                    'lower_bounds': result['lower_bounds'].tolist(),
                    'upper_bounds': result['upper_bounds'].tolist(),
                    'ibp_values': result['ibp_values'].tolist(),
                    'historical_dates': historical_dates,
                    'historical_values': result['historical_values'].tolist(),
                    'metrics': {
                        'chronos': {
                            'mae': float(chronos_mae),
                            'rmse': float(chronos_rmse),
                            'mape': float(chronos_mape) if chronos_mape is not None else None
                        },
                        'ibp': {
                            'mae': float(ibp_mae) if ibp_mae is not None else None,
                            'rmse': float(ibp_rmse) if ibp_rmse is not None else None,
                            'mape': float(ibp_mape) if ibp_mape is not None else None
                        },
                        'winner': winner
                    }
                }
    
    # 添加元数据
    metadata = {
        'input_date_range': f"{input_start} to {input_end}",
        'prediction_date_range': f"{predict_start} to {predict_end}",
        'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_devices': len(device_types),
        'processed_devices': overall_metrics['processed_devices'],
        'summary': {
            'chronos_wins': overall_metrics['chronos_wins'],
            'ibp_wins': overall_metrics['ibp_wins'],
            'overall_winner': "Chronos" if overall_metrics['chronos_wins'] > overall_metrics['ibp_wins'] else 
                              "IBP" if overall_metrics['ibp_wins'] > overall_metrics['chronos_wins'] else "Tie"
        }
    }
    
    # 创建完整的输出对象
    output_data = {
        'metadata': metadata,
        'results': serializable_results
    }
    
    # 生成文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    input_range = f"{input_start}_to_{input_end}"
    predict_range = f"{predict_start}_to_{predict_end}"
    filename = f"forecast_results_{input_range}_predict_{predict_range}_{timestamp}.json"
    
    # 保存到指定文件夹
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"已保存预测结果：{filename}")
    return output_path

def save_results_to_csv(results_dict, device_types, output_dir, input_start, input_end, predict_start, predict_end):
    """
    将预测结果保存为CSV文件
    
    参数:
    results_dict : dict, 包含每个设备预测结果的字典 {device_type: results}
    device_types : list, 设备类型列表
    output_dir : str, 输出目录
    input_start : str, 输入数据的开始日期
    input_end : str, 输入数据的结束日期
    predict_start : str, 预测的开始日期
    predict_end : str, 预测的结束日期
    
    返回:
    str: CSV文件路径
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 准备CSV数据
    csv_data = []
    
    # 生成表头
    header = ["设备类型", "日期", "实际值", "Chronos预测", "预测下界", "预测上界", "IBP预测", 
              "Chronos_MAE", "Chronos_RMSE", "Chronos_MAPE", 
              "IBP_MAE", "IBP_RMSE", "IBP_MAPE", "胜出方"]
    
    # 处理每个设备的结果
    for device_type in device_types:
        if device_type in results_dict:
            result = results_dict[device_type]
            
            # 计算评估指标
            valid_indices = ~np.isnan(result['actuals'])
            chronos_mae = None
            chronos_rmse = None
            chronos_mape = None
            ibp_mae = None
            ibp_rmse = None
            ibp_mape = None
            winner = "N/A"
            
            if np.any(valid_indices):
                # 计算Chronos的MAE和RMSE
                chronos_mae = np.mean(np.abs(result['actuals'][valid_indices] - result['predictions'][valid_indices]))
                chronos_rmse = np.sqrt(np.mean((result['actuals'][valid_indices] - result['predictions'][valid_indices])**2))
                
                # 计算Chronos的MAPE
                valid_mape_indices = valid_indices & (result['actuals'] != 0)
                if np.any(valid_mape_indices):
                    chronos_mape = np.mean(np.abs((result['actuals'][valid_mape_indices] - 
                                                result['predictions'][valid_mape_indices]) / 
                                                result['actuals'][valid_mape_indices])) * 100
                
                # 计算IBP的评估指标
                valid_ibp_indices = ~np.isnan(result['ibp_values'])
                if np.any(valid_ibp_indices):
                    ibp_mae = np.mean(np.abs(result['actuals'][valid_ibp_indices] - result['ibp_values'][valid_ibp_indices]))
                    ibp_rmse = np.sqrt(np.mean((result['actuals'][valid_ibp_indices] - result['ibp_values'][valid_ibp_indices])**2))
                    
                    # 计算IBP的MAPE
                    valid_ibp_mape_indices = valid_ibp_indices & (result['actuals'] != 0)
                    if np.any(valid_ibp_mape_indices):
                        ibp_mape = np.mean(np.abs((result['actuals'][valid_ibp_mape_indices] - 
                                                 result['ibp_values'][valid_ibp_mape_indices]) / 
                                                 result['actuals'][valid_ibp_mape_indices])) * 100
            
                # 判断胜出方
                if chronos_mae is not None and ibp_mae is not None:
                    chronos_wins = 0
                    ibp_wins = 0
                    
                    if chronos_mae < ibp_mae:
                        chronos_wins += 1
                    elif ibp_mae < chronos_mae:
                        ibp_wins += 1
                        
                    if chronos_rmse < ibp_rmse:
                        chronos_wins += 1
                    elif ibp_rmse < chronos_rmse:
                        ibp_wins += 1
                        
                    if chronos_mape is not None and ibp_mape is not None:
                        if chronos_mape < ibp_mape:
                            chronos_wins += 1
                        elif ibp_mape < chronos_mape:
                            ibp_wins += 1
                    
                    if chronos_wins >= 2:
                        winner = "Chronos"
                    elif ibp_wins >= 2:
                        winner = "IBP"
            
            # 为每个日期添加一行数据
            for i, date in enumerate(result['dates']):
                row = [
                    device_type,
                    date.strftime('%Y-%m-%d'),
                    result['actuals'][i] if not np.isnan(result['actuals'][i]) else "",
                    result['predictions'][i] if not np.isnan(result['predictions'][i]) else "",
                    result['lower_bounds'][i] if not np.isnan(result['lower_bounds'][i]) else "",
                    result['upper_bounds'][i] if not np.isnan(result['upper_bounds'][i]) else "",
                    result['ibp_values'][i] if i < len(result['ibp_values']) and not np.isnan(result['ibp_values'][i]) else "",
                    chronos_mae if chronos_mae is not None else "",
                    chronos_rmse if chronos_rmse is not None else "",
                    chronos_mape if chronos_mape is not None else "",
                    ibp_mae if ibp_mae is not None else "",
                    ibp_rmse if ibp_rmse is not None else "",
                    ibp_mape if ibp_mape is not None else "",
                    winner
                ]
                csv_data.append(row)
    
    # 生成文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    input_range = f"{input_start}_to_{input_end}"
    predict_range = f"{predict_start}_to_{predict_end}"
    filename = f"forecast_results_{input_range}_predict_{predict_range}_{timestamp}.csv"
    
    # 保存到CSV文件
    output_path = os.path.join(output_dir, filename)
    
    try:
        # 创建DataFrame并保存为CSV
        df = pd.DataFrame(csv_data, columns=header)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')  # 使用utf-8-sig以支持中文Excel打开
        logger.info(f"已保存CSV格式预测结果：{filename}")
        return output_path
    except Exception as e:
        logger.error(f"保存CSV文件时出错: {e}")
        return None

def save_results_to_excel(results_dict, device_types, output_dir, input_start, input_end, predict_start, predict_end):
    """
    将预测结果保存为Excel文件
    
    参数:
    results_dict : dict, 包含每个设备预测结果的字典 {device_type: results}
    device_types : list, 设备类型列表
    output_dir : str, 输出目录
    input_start : str, 输入数据的开始日期
    input_end : str, 输入数据的结束日期
    predict_start : str, 预测的开始日期
    predict_end : str, 预测的结束日期
    
    返回:
    str: Excel文件路径
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取所有预测月份
    predict_months = []
    predict_start_date = pd.to_datetime(predict_start)
    predict_end_date = pd.to_datetime(predict_end)
    current = predict_start_date
    while current <= predict_end_date:
        predict_months.append(current.strftime('%b').upper() + ' ' + current.strftime('%Y'))
        current = current + pd.DateOffset(months=1)
    
    # 创建结果DataFrame的列名
    columns = ['计划机型', '关键指标', 'MAE'] + predict_months
    result_data = []
    
    # 用于统计摘要信息
    overall_metrics = {
        'processed_devices': 0,
        'chronos_wins': 0,
        'ibp_wins': 0
    }
    
    # 处理每个设备的结果
    for device_type in device_types:
        if device_type in results_dict:
            result = results_dict[device_type]
            overall_metrics['processed_devices'] += 1
            
            # 计算评估指标
            valid_indices = ~np.isnan(result['actuals'])
            chronos_mae = None
            chronos_rmse = None
            chronos_mape = None
            ibp_mae = None
            ibp_rmse = None
            ibp_mape = None
            winner = "N/A"
            
            if np.any(valid_indices):
                # 计算Chronos的MAE和RMSE
                chronos_mae = np.mean(np.abs(result['actuals'][valid_indices] - result['predictions'][valid_indices]))
                chronos_rmse = np.sqrt(np.mean((result['actuals'][valid_indices] - result['predictions'][valid_indices])**2))
                
                # 计算Chronos的MAPE
                valid_mape_indices = valid_indices & (result['actuals'] != 0)
                if np.any(valid_mape_indices):
                    chronos_mape = np.mean(np.abs((result['actuals'][valid_mape_indices] - 
                                                result['predictions'][valid_mape_indices]) / 
                                                result['actuals'][valid_mape_indices])) * 100
                
                # 计算IBP的评估指标
                valid_ibp_indices = ~np.isnan(result['ibp_values'])
                if np.any(valid_ibp_indices):
                    ibp_mae = np.mean(np.abs(result['actuals'][valid_ibp_indices] - result['ibp_values'][valid_ibp_indices]))
                    ibp_rmse = np.sqrt(np.mean((result['actuals'][valid_ibp_indices] - result['ibp_values'][valid_ibp_indices])**2))
                    
                    # 计算IBP的MAPE
                    valid_ibp_mape_indices = valid_ibp_indices & (result['actuals'] != 0)
                    if np.any(valid_ibp_mape_indices):
                        ibp_mape = np.mean(np.abs((result['actuals'][valid_ibp_mape_indices] - 
                                                 result['ibp_values'][valid_ibp_mape_indices]) / 
                                                 result['actuals'][valid_ibp_mape_indices])) * 100
            
                # 判断胜出方
                if chronos_mae is not None and ibp_mae is not None:
                    chronos_wins = 0
                    ibp_wins = 0
                    
                    if chronos_mae < ibp_mae:
                        chronos_wins += 1
                    elif ibp_mae < chronos_mae:
                        ibp_wins += 1
                        
                    if chronos_rmse < ibp_rmse:
                        chronos_wins += 1
                    elif ibp_rmse < chronos_rmse:
                        ibp_wins += 1
                        
                    if chronos_mape is not None and ibp_mape is not None:
                        if chronos_mape < ibp_mape:
                            chronos_wins += 1
                        elif ibp_mape < chronos_mape:
                            ibp_wins += 1
                    
                    if chronos_wins >= 2:
                        winner = "Chronos"
                        overall_metrics['chronos_wins'] += 1
                    elif ibp_wins >= 2:
                        winner = "IBP"
                        overall_metrics['ibp_wins'] += 1
            
            # 创建Chronos预测值行
            chronos_row = {
                '计划机型': device_type,
                '关键指标': 'Chronos预测值',
                'MAE': round(float(chronos_mae), 2) if chronos_mae is not None else "N/A"
            }
            
            # 创建IBP预测值行
            ibp_row = {
                '计划机型': device_type,
                '关键指标': 'IBP预测值',
                'MAE': round(float(ibp_mae), 2) if ibp_mae is not None else "N/A"
            }
            
            # 创建实际发货数量行
            actual_row = {
                '计划机型': device_type,
                '关键指标': '实际发货数量',
                'MAE': ""  # 实际值行不显示MAE
            }
            
            # 为每个预测月份添加数据
            for i, date in enumerate(result['dates']):
                month_col = date.strftime('%b').upper() + ' ' + date.strftime('%Y')
                
                # Chronos预测值
                if i < len(result['predictions']):
                    chronos_row[month_col] = round(float(result['predictions'][i]))
                
                # IBP预测值
                if i < len(result['ibp_values']):
                    ibp_value = result['ibp_values'][i]
                    ibp_row[month_col] = round(float(ibp_value)) if not np.isnan(ibp_value) else "N/A"
                
                # 实际值
                if i < len(result['actuals']):
                    actual_row[month_col] = int(result['actuals'][i]) if not np.isnan(result['actuals'][i]) else 0
            
            # 添加三行到结果数据
            result_data.append(chronos_row)
            result_data.append(ibp_row)
            result_data.append(actual_row)
    
    # 生成文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    input_range = f"{input_start}_to_{input_end}"
    predict_range = f"{predict_start}_to_{predict_end}"
    filename = f"forecast_results_{input_range}_predict_{predict_range}_{timestamp}.xlsx"
    
    # 保存到Excel文件
    output_path = os.path.join(output_dir, filename)
    
    try:
        # 创建DataFrame
        df = pd.DataFrame(result_data)
        
        # 确保所有列都存在，以正确的顺序
        for col in columns:
            if col not in df.columns:
                df[col] = ""
        df = df[columns]  # 按需重新排序列
        
        # 创建一个Excel写入器
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 将数据写入主工作表
            df.to_excel(writer, sheet_name='预测结果', index=False)
            
            # 创建摘要工作表
            summary_data = {
                '类别': ['输入日期范围', '预测日期范围', '生成时间', '设备总数', '处理的设备数'],
                '值': [
                    f"{input_start} 到 {input_end}",
                    f"{predict_start} 到 {predict_end}",
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    len(device_types),
                    overall_metrics['processed_devices']
                ]
            }
            
            # 添加胜出统计到摘要
            summary_data['类别'].extend(['Chronos胜出次数', 'IBP胜出次数', '总体胜出方'])
            summary_data['值'].extend([
                overall_metrics['chronos_wins'],
                overall_metrics['ibp_wins'],
                "Chronos" if overall_metrics['chronos_wins'] > overall_metrics['ibp_wins'] else 
                "IBP" if overall_metrics['ibp_wins'] > overall_metrics['chronos_wins'] else "平局"
            ])
            
            # 写入摘要工作表
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='摘要', index=False)
            
        logger.info(f"已保存Excel格式预测结果：{filename}")
        return output_path
    except Exception as e:
        logger.error(f"保存Excel文件时出错: {e}")
        return None

def analyze_multiple_devices(excel_path, device_types, input_start='2023-03', input_end='2023-12', 
                          predict_start='2024-01', predict_end='2024-12', memory_efficient=True):
    """
    分析多个设备类型并保存预测结果为JSON和Excel文件
    
    参数:
    excel_path : str, Excel文件路径
    device_types : list, 设备类型列表
    input_start : str, 输入数据的开始日期 (格式: 'YYYY-MM')
    input_end : str, 输入数据的结束日期 (格式: 'YYYY-MM')
    predict_start : str, 预测的开始日期 (格式: 'YYYY-MM')
    predict_end : str, 预测的结束日期 (格式: 'YYYY-MM')
    memory_efficient : bool, 是否使用内存高效模式 (默认True)
    
    返回:
    dict: 包含JSON和Excel文件路径的字典
    """
    # 设置日期范围
    logger.info("正在设置日期范围...")
    date_ranges = setup_date_ranges(input_start, input_end, predict_start, predict_end)
    
    # 初始化Chronos模型
    logger.info("正在加载Chronos模型，这可能需要几分钟时间...")
    try:
        # 内存高效模式设置
        device_map = "auto" if memory_efficient else "cuda"
        dtype = torch.float16 if memory_efficient else torch.bfloat16
        
        pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-t5-base",
            device_map=device_map,
            torch_dtype=dtype,
        )
        logger.info("Chronos模型加载成功!")
    except Exception as e:
        logger.error(f"加载模型时出错: {e}")
        logger.info("尝试使用CPU加载模型...")
        pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-t5-base",
            device_map="cpu",
        )
        logger.info("使用CPU加载模型成功!")
    
    # 读取Excel数据
    logger.info(f"正在读取Excel数据: {excel_path}...")
    try:
        # 检查文件是否存在
        if not os.path.exists(excel_path):
            logger.error(f"错误: Excel文件不存在: {excel_path}")
            logger.info(f"当前工作目录: {os.getcwd()}")
            logger.info(f"目录中的文件: {os.listdir('.')}")
            return None
            
        # 尝试读取Excel文件
        df = pd.read_excel(excel_path)
        logger.info(f"Excel数据读取成功，共 {len(df)} 行")
        
        # 检查必要的列
        if "计划机型" not in df.columns:
            logger.error(f"错误：Excel文件缺少必要的列 '计划机型'")
            logger.info(f"文件中的列: {df.columns.tolist()}")
            return None
            
    except Exception as e:
        logger.error(f"读取Excel文件时出错: {e}")
        return None
    
    # 创建输出文件夹
    output_dir = "forecast_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"创建输出目录: {output_dir}")
    
    # 存储所有设备的预测结果
    results_dict = {}
    total_devices = len(device_types)
    
    # 处理每个设备类型
    logger.info(f"开始处理 {total_devices} 个设备...")
    
    # 添加总进度条
    progress_bar = tqdm(total=total_devices, desc="设备处理进度", ncols=100)
    
    for i, device_type in enumerate(device_types):
        # 更新进度条描述
        progress_bar.set_description(f"正在处理: {device_type}")
        
        logger.debug(f"处理设备 {i+1}/{total_devices}: {device_type}...")
        # 过滤和处理数据
        device_df = filter_device_data(df, device_type, date_ranges)
        if device_df is not None:
            logger.debug(f"设备 {device_type} 数据有效，进行预测分析")
            # 分析数据
            results = analyze_device(device_df, pipeline, device_type, date_ranges)
            results_dict[device_type] = results
        else:
            logger.debug(f"设备 {device_type} 数据无效，跳过")
        
        # 更新进度条
        progress_bar.update(1)
    
    # 关闭进度条
    progress_bar.close()
    
    # 将结果保存为JSON和Excel
    logger.info("所有设备处理完成，正在保存结果...")
    json_path = save_results_to_json(results_dict, device_types, output_dir, input_start, input_end, predict_start, predict_end)
    excel_path = save_results_to_excel(results_dict, device_types, output_dir, input_start, input_end, predict_start, predict_end)
    
    return {"json": json_path, "excel": excel_path}

def analyze_sample(excel_path, sample_count=3, input_start='2023-03', input_end='2023-12', 
                  predict_start='2024-01', predict_end='2024-12', memory_efficient=True):
    """
    分析样本设备以快速测试流程
    
    参数:
    excel_path : str, Excel文件路径
    sample_count : int, 要处理的样本设备数量
    input_start : str, 输入数据的开始日期 (格式: 'YYYY-MM')
    input_end : str, 输入数据的结束日期 (格式: 'YYYY-MM')
    predict_start : str, 预测的开始日期 (格式: 'YYYY-MM')
    predict_end : str, 预测的结束日期 (格式: 'YYYY-MM')
    memory_efficient : bool, 是否使用内存高效模式 (默认True)
    
    返回:
    dict: 包含JSON和Excel文件路径的字典
    """
    device_types = [f"Model{i}" for i in range(1, sample_count+1)]
    logger.info(f"测试模式: 仅处理 {sample_count} 个设备 {device_types}")
    return analyze_multiple_devices(excel_path, device_types, input_start, input_end, predict_start, predict_end, memory_efficient)

def get_device_types_from_excel(excel_path):
    """
    从Excel文件中读取所有实际存在的设备类型
    
    参数:
    excel_path : str, Excel文件路径
    
    返回:
    list: 设备类型列表
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(excel_path):
            logger.error(f"错误: Excel文件不存在: {excel_path}")
            return None
            
        # 读取Excel文件
        df = pd.read_excel(excel_path)
        logger.info(f"Excel数据读取成功，共 {len(df)} 行")
        
        # 检查必要的列
        if "计划机型" not in df.columns:
            logger.error(f"错误：Excel文件缺少必要的列 '计划机型'")
            return None
            
        # 获取唯一的设备类型
        device_types = df["计划机型"].unique().tolist()
        logger.info(f"从Excel中提取了 {len(device_types)} 个不同的设备类型")
        return device_types
        
    except Exception as e:
        logger.error(f"读取Excel设备类型时出错: {e}")
        return None

if __name__ == "__main__":
    print("Chronos预测系统启动...")
    print("所有日志将记录到日志文件，控制台将仅显示进度条")
    
    excel_path = "IBP统计预测和实际发货值数据-20250311-转换后.xlsx"
    
    # 默认使用2023-03到2023-12作为输入，预测2024-01到2024-12
    input_start = '2023-03'
    input_end = '2023-12'
    predict_start = '2024-01' 
    predict_end = '2024-12'
    
    # 使用高内存效率模式
    memory_efficient = True
    
    # 创建总体进度条
    overall_progress = tqdm(total=4, desc="总体进度", position=0, ncols=100)
    
    # 测试模式: 只处理少量设备快速验证程序是否能运行
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        logger.info("正在测试模式下运行程序...")
        
        overall_progress.set_description("测试模式: 加载数据")
        overall_progress.update(1)
        
        overall_progress.set_description("测试模式: 加载模型")
        overall_progress.update(1)
        
        overall_progress.set_description("测试模式: 处理设备")
        result_paths = analyze_sample(excel_path, sample_count=3, 
                                     input_start=input_start, input_end=input_end, 
                                     predict_start=predict_start, predict_end=predict_end,
                                     memory_efficient=memory_efficient)
        overall_progress.update(1)
        
        overall_progress.set_description("测试模式: 保存结果")
        overall_progress.update(1)
    else:
        # 完整处理: 处理所有设备
        logger.info("正在完整模式下运行程序...")
        
        overall_progress.set_description("完整模式: 加载数据")
        overall_progress.update(1)
        
        # 从Excel文件中自动获取设备类型
        device_types = get_device_types_from_excel(excel_path)
        
        if not device_types:
            logger.info("无法获取设备类型列表，使用默认范围...")
            device_types = [f"Model{i}" for i in range(1, 300)]  # 默认处理Model1到Model299
        
        overall_progress.set_description("完整模式: 加载模型")    
        overall_progress.update(1)
        
        logger.info(f"将处理 {len(device_types)} 个设备")
        
        overall_progress.set_description("完整模式: 处理设备")
        result_paths = analyze_multiple_devices(excel_path, device_types, 
                                              input_start, input_end, predict_start, predict_end,
                                              memory_efficient=memory_efficient)
        overall_progress.update(1)
        
        overall_progress.set_description("完整模式: 保存结果")
        overall_progress.update(1)
    
    # 关闭总体进度条
    overall_progress.close()
    
    if result_paths and result_paths.get("json") and result_paths.get("excel"):
        logger.info(f"\n预测完成！结果已保存至：")
        logger.info(f"JSON文件: {result_paths['json']}")
        logger.info(f"Excel文件: {result_paths['excel']}")
        print(f"\n预测完成！结果已保存为JSON和Excel格式。")
    else:
        logger.error("\n预测失败，没有生成结果文件。请检查错误信息。")
        print("\n预测失败，没有生成结果文件。请检查日志了解详情。") 
