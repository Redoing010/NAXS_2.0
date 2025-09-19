import akshare as ak
import backtrader as bt
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 设置为非交互模式
import matplotlib.pyplot as plt
import logging
import sys
from typing import Optional
from datetime import datetime
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('akshare_backtrade.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 1. 使用 AKShare 获取数据
def fetch_data(symbol: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
    """
    获取期货历史数据
    
    Args:
        symbol: 期货代码
        max_retries: 最大重试次数
    
    Returns:
        处理后的DataFrame，失败时返回None
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"正在获取期货 {symbol} 的历史数据 (尝试 {attempt + 1}/{max_retries})")
            
            # 获取期货主力合约历史数据 (使用新浪财经接口)
            df = ak.futures_main_sina(symbol=symbol, start_date='20200101')
            
            if df is None or df.empty:
                raise ValueError(f"获取到的数据为空")
            
            # 数据预处理 - 处理中文列名
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)
            df.sort_index(inplace=True)
            
            # 重命名列为英文，便于backtrader使用
            column_mapping = {
                '开盘价': 'open',
                '最高价': 'high', 
                '最低价': 'low',
                '收盘价': 'close',
                '成交量': 'volume'
            }
            df.rename(columns=column_mapping, inplace=True)
            
            # 数据验证
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"缺少必要的数据列: {missing_columns}")
            
            # 检查数据质量
            if df['close'].isna().sum() > len(df) * 0.1:  # 如果超过10%的收盘价为空
                raise ValueError("数据质量不佳，过多空值")
            
            logger.info(f"成功获取 {len(df)} 条期货数据记录")
            return df
            
        except Exception as e:
            logger.warning(f"获取期货数据失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(2)  # 重试前等待2秒
            else:
                logger.error(f"获取期货数据最终失败: {e}")
                return None

# 2. 创建增强的Backtrader策略
class EnhancedStrategy(bt.Strategy):
    """
    增强的交易策略，包含多个技术指标和风险管理
    """
    params = (
        ('ma_period', 20),      # 移动平均线周期
        ('rsi_period', 14),     # RSI周期
        ('rsi_upper', 70),      # RSI超买线
        ('rsi_lower', 30),      # RSI超卖线
        ('stop_loss', 0.05),    # 止损比例 (5%)
        ('take_profit', 0.10),  # 止盈比例 (10%)
    )

    def __init__(self):
        # 记录收盘价数据
        self.dataclose = self.datas[0].close
        
        # 技术指标
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.ma_period
        )
        self.rsi = bt.indicators.RSI(
            self.datas[0], period=self.params.rsi_period
        )
        
        # 交易记录
        self.order = None
        self.buy_price = None
        self.buy_comm = None
        
        # 统计信息
        self.trade_count = 0
        self.win_count = 0
        
        logger.info("策略初始化完成")
        logger.info(f"参数: MA周期={self.params.ma_period}, RSI周期={self.params.rsi_period}")

    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                logger.info(f"买入执行: 价格={order.executed.price:.2f}, 成本={order.executed.value:.2f}, 手续费={order.executed.comm:.2f}")
                self.buy_price = order.executed.price
                self.buy_comm = order.executed.comm
            else:
                logger.info(f"卖出执行: 价格={order.executed.price:.2f}, 成本={order.executed.value:.2f}, 手续费={order.executed.comm:.2f}")
                
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.warning(f"订单失败: {order.status}")

        self.order = None

    def notify_trade(self, trade):
        """交易完成通知"""
        if not trade.isclosed:
            return

        self.trade_count += 1
        profit = trade.pnl
        
        if profit > 0:
            self.win_count += 1
            logger.info(f"✅ 盈利交易: {profit:.2f} (净利润: {trade.pnlcomm:.2f})")
        else:
            logger.info(f"❌ 亏损交易: {profit:.2f} (净利润: {trade.pnlcomm:.2f})")
        
        win_rate = (self.win_count / self.trade_count) * 100 if self.trade_count > 0 else 0
        logger.info(f"交易统计: 总交易={self.trade_count}, 胜率={win_rate:.1f}%")

    def next(self):
        """策略主逻辑"""
        # 如果有未完成的订单，跳过
        if self.order:
            return

        # 获取当前数据
        current_price = self.dataclose[0]
        sma_value = self.sma[0]
        rsi_value = self.rsi[0]
        
        # 风险管理：止损止盈
        if self.position:
            # 计算盈亏比例
            pnl_pct = (current_price - self.buy_price) / self.buy_price
            
            # 止损
            if pnl_pct <= -self.params.stop_loss:
                logger.warning(f"触发止损: 当前价格={current_price:.2f}, 买入价格={self.buy_price:.2f}, 亏损={pnl_pct:.2%}")
                self.order = self.sell()
                return
            
            # 止盈
            if pnl_pct >= self.params.take_profit:
                logger.info(f"触发止盈: 当前价格={current_price:.2f}, 买入价格={self.buy_price:.2f}, 盈利={pnl_pct:.2%}")
                self.order = self.sell()
                return
            
            # 技术指标卖出信号
            if (current_price < sma_value and rsi_value > self.params.rsi_upper):
                logger.info(f"技术指标卖出: 价格={current_price:.2f}, SMA={sma_value:.2f}, RSI={rsi_value:.1f}")
                self.order = self.sell()
        
        else:
            # 买入信号：价格上穿均线且RSI不在超买区域
            if (current_price > sma_value and 
                rsi_value < self.params.rsi_upper and 
                rsi_value > self.params.rsi_lower):
                logger.info(f"买入信号: 价格={current_price:.2f}, SMA={sma_value:.2f}, RSI={rsi_value:.1f}")
                self.order = self.buy()

def check_dependencies() -> bool:
    """
    检查必要的依赖包是否已安装
    
    Returns:
        如果所有依赖都可用返回True，否则返回False
    """
    required_packages = ['akshare', 'backtrader', 'pandas', 'matplotlib']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"缺少必要的依赖包: {missing_packages}")
        logger.error("请运行: pip install -r requirements.txt")
        return False
    
    return True

def run_backtest(symbol: str = 'RB0', initial_cash: float = 100000.0) -> None:
    """
    运行回测
    
    Args:
        symbol: 期货代码
        initial_cash: 初始资金
    """
    logger.info("=" * 50)
    logger.info("AKShare期货回测系统启动")
    logger.info("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    try:
        # 获取数据
        logger.info(f"正在获取期货 {symbol} 的历史数据...")
        data = fetch_data(symbol)
        
        if data is None:
            logger.error("无法获取数据，回测终止")
            return
        
        logger.info(f"数据获取成功，时间范围: {data.index[0]} 到 {data.index[-1]}")
        
        # 初始化cerebro回测引擎
        cerebro = bt.Cerebro()
        
        # 添加策略
        cerebro.addstrategy(EnhancedStrategy)
        
        # 将Pandas DataFrame转换为Backtrader的数据馈送
        data_feed = bt.feeds.PandasData(dataname=data)
        cerebro.adddata(data_feed)
        
        # 设置初始资金
        cerebro.broker.setcash(initial_cash)
        
        # 设置手续费（期货手续费通常较高）
        cerebro.broker.setcommission(commission=0.0005)
        
        # 添加分析器
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        
        # 运行回测
        logger.info(f"开始回测，初始资金: {initial_cash:,.2f}")
        start_time = datetime.now()
        
        results = cerebro.run()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 获取结果
        final_value = cerebro.broker.getvalue()
        total_return = (final_value - initial_cash) / initial_cash * 100
        
        # 获取分析结果
        strat = results[0]
        sharpe_ratio = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
        max_drawdown = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0)
        
        # 输出结果
        logger.info("=" * 50)
        logger.info("回测结果")
        logger.info("=" * 50)
        logger.info(f"初始资金: {initial_cash:,.2f}")
        logger.info(f"最终资金: {final_value:,.2f}")
        logger.info(f"总收益率: {total_return:.2f}%")
        logger.info(f"夏普比率: {sharpe_ratio:.3f}" if sharpe_ratio else "夏普比率: N/A")
        logger.info(f"最大回撤: {max_drawdown:.2f}%")
        logger.info(f"回测耗时: {duration:.2f}秒")
        
        # 绘制回测结果图并保存
        try:
            logger.info("正在生成回测图表...")
            fig = cerebro.plot(style='candlestick', barup='red', bardown='green', returnfig=True)[0][0]
            
            # 保存图表到文件
            chart_filename = f"backtest_chart_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            fig.savefig(chart_filename, dpi=300, bbox_inches='tight')
            plt.close(fig)  # 关闭图形以释放内存
            
            logger.info(f"回测完成！图表已保存为: {chart_filename}")
        except Exception as e:
            logger.warning(f"图表生成失败: {e}")
        
    except Exception as e:
        logger.error(f"回测过程中发生错误: {e}")
        raise

# 3. 主程序入口
if __name__ == '__main__':
    # 可以通过以下代码查看akshare期货相关函数（调试时使用）
    # print([attr for attr in dir(ak) if 'futures' in attr.lower()])
    
    try:
        # 运行回测，可以修改参数
        run_backtest(
            symbol='RB0',      # 螺纹钢主力合约
            initial_cash=100000.0  # 10万初始资金
        )
    except KeyboardInterrupt:
        logger.info("用户中断程序")
    except Exception as e:
        logger.error(f"程序运行失败: {e}")
        sys.exit(1)