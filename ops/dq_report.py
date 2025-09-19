#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据质量报告生成脚本

用法:
    python ops/dq_report.py --start 2024-01-01 --end 2024-12-31 \
        --freq D --parquet-root data/parquet --out reports/dq/latest
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
import json

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.data.dq import DataQualityChecker, check_data_quality
from modules.data.parquet_store import ParquetStore
from modules.data.utils import normalize_symbol

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/dq_report.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='生成数据质量报告')
    
    parser.add_argument('--start', type=str, required=True,
                       help='开始日期 YYYY-MM-DD')
    
    parser.add_argument('--end', type=str, required=True,
                       help='结束日期 YYYY-MM-DD')
    
    parser.add_argument('--freq', type=str, default='D',
                       choices=['D'],
                       help='数据频率')
    
    parser.add_argument('--parquet-root', type=str, default='data/parquet',
                       help='Parquet数据根目录')
    
    parser.add_argument('--out', type=str, default='reports/dq/latest',
                       help='报告输出目录')
    
    parser.add_argument('--symbols', type=str, nargs='*',
                       help='指定股票代码列表，不指定则使用所有可用股票')
    
    parser.add_argument('--symbols-file', type=str,
                       help='从文件读取股票代码列表（每行一个）')
    
    parser.add_argument('--market', type=str, default='all',
                       choices=['all', 'sh', 'sz'],
                       help='市场过滤')
    
    parser.add_argument('--sample-size', type=int, default=0,
                       help='随机采样数量，0表示全部')
    
    parser.add_argument('--format', type=str, default='json',
                       choices=['json', 'html', 'markdown'],
                       help='报告格式')
    
    parser.add_argument('--rules', type=str, nargs='*',
                       help='指定要执行的规则名称')
    
    parser.add_argument('--disable-rules', type=str, nargs='*',
                       help='禁用的规则名称')
    
    parser.add_argument('--severity', type=str, default='all',
                       choices=['all', 'error', 'warning', 'info'],
                       help='问题严重程度过滤')
    
    parser.add_argument('--parallel', type=int, default=1,
                       help='并行处理数量')
    
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    
    return parser.parse_args()


def setup_logging(log_level: str):
    """设置日志级别"""
    level = getattr(logging, log_level.upper())
    logging.getLogger().setLevel(level)
    
    # 确保日志目录存在
    Path('logs').mkdir(exist_ok=True)


def load_symbols_from_file(file_path: str) -> List[str]:
    """从文件加载股票代码列表"""
    symbols = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    symbols.append(line)
        logger.info(f"从文件加载 {len(symbols)} 个股票代码: {file_path}")
        return symbols
    except Exception as e:
        logger.error(f"加载股票代码文件失败: {e}")
        return []


def get_available_symbols(parquet_root: str, market: str = 'all') -> List[str]:
    """获取可用的股票代码列表"""
    try:
        store = ParquetStore()
        all_symbols = store.list_symbols(parquet_root)
        
        if market == 'all':
            return all_symbols
        elif market == 'sh':
            return [s for s in all_symbols if s.endswith('.SH')]
        elif market == 'sz':
            return [s for s in all_symbols if s.endswith('.SZ')]
        else:
            return all_symbols
            
    except Exception as e:
        logger.error(f"获取可用股票代码失败: {e}")
        return []


def sample_symbols(symbols: List[str], sample_size: int) -> List[str]:
    """随机采样股票代码"""
    if sample_size <= 0 or sample_size >= len(symbols):
        return symbols
    
    import random
    sampled = random.sample(symbols, sample_size)
    logger.info(f"随机采样 {sample_size} 个股票代码")
    return sampled


def generate_html_report(summary: Dict[str, Any], reports: Dict[str, Any], 
                        output_file: str):
    """生成HTML格式报告"""
    try:
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>数据质量报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: #fff; border: 1px solid #ddd; padding: 15px; border-radius: 5px; flex: 1; }}
        .error {{ color: #d32f2f; }}
        .warning {{ color: #f57c00; }}
        .success {{ color: #388e3c; }}
        .issue-list {{ margin: 20px 0; }}
        .issue-item {{ background: #f9f9f9; margin: 10px 0; padding: 15px; border-radius: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f5f5f5; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>数据质量报告</h1>
        <p>生成时间: {summary['generated_at']}</p>
    </div>
    
    <div class="summary">
        <h2>概览</h2>
        <div class="stats">
            <div class="stat-card">
                <h3>总股票数</h3>
                <p style="font-size: 24px; margin: 0;">{summary['total_symbols']}</p>
            </div>
            <div class="stat-card">
                <h3 class="success">健康股票</h3>
                <p style="font-size: 24px; margin: 0;">{summary['healthy_symbols']}</p>
            </div>
            <div class="stat-card">
                <h3 class="error">有错误股票</h3>
                <p style="font-size: 24px; margin: 0;">{summary['symbols_with_errors']}</p>
            </div>
            <div class="stat-card">
                <h3 class="warning">有警告股票</h3>
                <p style="font-size: 24px; margin: 0;">{summary['symbols_with_warnings']}</p>
            </div>
        </div>
    </div>
    
    <div class="issue-breakdown">
        <h2>问题分类统计</h2>
        <table>
            <tr><th>问题类型</th><th>数量</th></tr>
"""
        
        for issue_type, count in summary['issue_breakdown'].items():
            html_content += f"            <tr><td>{issue_type}</td><td>{count}</td></tr>\n"
        
        html_content += """
        </table>
    </div>
    
    <div class="top-issues">
        <h2>严重问题</h2>
        <div class="issue-list">
"""
        
        for issue in summary['top_issues'][:10]:
            html_content += f"""
            <div class="issue-item">
                <h4 class="error">{issue['rule_name']}</h4>
                <p><strong>股票:</strong> {issue['symbol']}</p>
                <p><strong>消息:</strong> {issue['message']}</p>
                <p><strong>时间:</strong> {issue['timestamp']}</p>
            </div>
"""
        
        html_content += """
        </div>
    </div>
    
</body>
</html>
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML报告已生成: {output_file}")
        
    except Exception as e:
        logger.error(f"生成HTML报告失败: {e}")


def generate_markdown_report(summary: Dict[str, Any], reports: Dict[str, Any], 
                           output_file: str):
    """生成Markdown格式报告"""
    try:
        md_content = f"""
# 数据质量报告

**生成时间:** {summary['generated_at']}

## 概览

| 指标 | 数量 |
|------|------|
| 总股票数 | {summary['total_symbols']} |
| 健康股票 | {summary['healthy_symbols']} |
| 有错误股票 | {summary['symbols_with_errors']} |
| 有警告股票 | {summary['symbols_with_warnings']} |
| 总问题数 | {summary['total_issues']} |

## 问题分类统计

| 问题类型 | 数量 |
|----------|------|
"""
        
        for issue_type, count in summary['issue_breakdown'].items():
            md_content += f"| {issue_type} | {count} |\n"
        
        md_content += "\n## 严重问题\n\n"
        
        for i, issue in enumerate(summary['top_issues'][:10], 1):
            md_content += f"""
### {i}. {issue['rule_name']}

- **股票:** {issue['symbol']}
- **消息:** {issue['message']}
- **时间:** {issue['timestamp']}

"""
        
        # 添加详细统计
        md_content += "\n## 详细统计\n\n"
        
        healthy_rate = summary['healthy_symbols'] / summary['total_symbols'] * 100 if summary['total_symbols'] > 0 else 0
        md_content += f"- **健康率:** {healthy_rate:.1f}%\n"
        
        error_rate = summary['symbols_with_errors'] / summary['total_symbols'] * 100 if summary['total_symbols'] > 0 else 0
        md_content += f"- **错误率:** {error_rate:.1f}%\n"
        
        warning_rate = summary['symbols_with_warnings'] / summary['total_symbols'] * 100 if summary['total_symbols'] > 0 else 0
        md_content += f"- **警告率:** {warning_rate:.1f}%\n"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"Markdown报告已生成: {output_file}")
        
    except Exception as e:
        logger.error(f"生成Markdown报告失败: {e}")


def filter_reports_by_severity(reports: Dict[str, Any], severity: str) -> Dict[str, Any]:
    """根据严重程度过滤报告"""
    if severity == 'all':
        return reports
    
    filtered_reports = {}
    
    for symbol, report in reports.items():
        filtered_issues = []
        for issue in report.issues:
            if issue.severity == severity:
                filtered_issues.append(issue)
        
        if filtered_issues:
            # 创建新的报告对象，只包含指定严重程度的问题
            from modules.data.dq import QualityReport
            filtered_report = QualityReport(
                symbol=report.symbol,
                date_range=report.date_range,
                total_records=report.total_records,
                issues=filtered_issues,
                metrics=report.metrics,
                generated_at=report.generated_at
            )
            filtered_reports[symbol] = filtered_report
    
    return filtered_reports


def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    logger.info(f"开始执行数据质量报告生成任务: {args}")
    
    # 验证日期范围
    try:
        start_dt = datetime.strptime(args.start, '%Y-%m-%d')
        end_dt = datetime.strptime(args.end, '%Y-%m-%d')
        
        if start_dt >= end_dt:
            logger.error(f"开始日期必须早于结束日期: {args.start} >= {args.end}")
            sys.exit(1)
            
    except ValueError as e:
        logger.error(f"日期格式错误: {e}")
        sys.exit(1)
    
    # 确定股票代码列表
    symbols = []
    
    if args.symbols:
        # 使用命令行指定的股票代码
        symbols = [normalize_symbol(s) for s in args.symbols]
        logger.info(f"使用指定的股票代码: {len(symbols)} 个")
    elif args.symbols_file:
        # 从文件读取股票代码
        symbols = load_symbols_from_file(args.symbols_file)
        if not symbols:
            logger.error("从文件读取股票代码失败")
            sys.exit(1)
    else:
        # 使用所有可用的股票代码
        symbols = get_available_symbols(args.parquet_root, args.market)
        if not symbols:
            logger.error("未找到可用的股票代码")
            sys.exit(1)
        logger.info(f"使用所有可用股票代码: {len(symbols)} 个")
    
    # 随机采样
    if args.sample_size > 0:
        symbols = sample_symbols(symbols, args.sample_size)
    
    logger.info(f"最终处理股票数量: {len(symbols)}")
    
    # 创建输出目录
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 初始化质量检查器
        checker = DataQualityChecker()
        
        # 禁用指定规则
        if args.disable_rules:
            for rule_name in args.disable_rules:
                checker.disable_rule(rule_name)
                logger.info(f"禁用规则: {rule_name}")
        
        # 执行质量检查
        logger.info("开始执行数据质量检查...")
        reports = checker.check_multiple_symbols(
            symbols, args.start, args.end, args.parquet_root
        )
        
        # 根据严重程度过滤
        if args.severity != 'all':
            reports = filter_reports_by_severity(reports, args.severity)
            logger.info(f"按严重程度过滤后剩余: {len(reports)} 个报告")
        
        # 生成汇总报告
        summary = checker.generate_summary_report(reports)
        
        # 保存报告
        if args.format == 'json':
            # JSON格式
            checker.save_summary_report(summary, output_dir / "summary.json")
            
            # 保存详细报告
            for symbol, report in reports.items():
                checker.save_report(report, output_dir / f"{symbol}.json")
                
        elif args.format == 'html':
            # HTML格式
            generate_html_report(summary, reports, output_dir / "report.html")
            
        elif args.format == 'markdown':
            # Markdown格式
            generate_markdown_report(summary, reports, output_dir / "report.md")
        
        # 打印汇总信息
        logger.info("=" * 50)
        logger.info("数据质量检查完成")
        logger.info(f"总股票数: {summary['total_symbols']}")
        logger.info(f"健康股票: {summary['healthy_symbols']}")
        logger.info(f"有错误股票: {summary['symbols_with_errors']}")
        logger.info(f"有警告股票: {summary['symbols_with_warnings']}")
        logger.info(f"总问题数: {summary['total_issues']}")
        
        if summary['total_symbols'] > 0:
            healthy_rate = summary['healthy_symbols'] / summary['total_symbols'] * 100
            logger.info(f"健康率: {healthy_rate:.1f}%")
        
        logger.info(f"报告已保存到: {output_dir}")
        logger.info("=" * 50)
        
        # 返回码
        if summary['symbols_with_errors'] > 0:
            logger.warning(f"发现 {summary['symbols_with_errors']} 个股票有错误")
            sys.exit(2)
        elif summary['symbols_with_warnings'] > 0:
            logger.warning(f"发现 {summary['symbols_with_warnings']} 个股票有警告")
            sys.exit(1)
        else:
            logger.info("所有股票数据质量良好")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.warning("用户中断操作")
        sys.exit(130)
    except Exception as e:
        logger.error(f"生成数据质量报告失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()