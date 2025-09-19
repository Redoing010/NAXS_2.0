# 任务规划器
# 解析用户意图并制定执行计划

import re
import json
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

class IntentType(Enum):
    """意图类型"""
    MARKET_ANALYSIS = "market_analysis"          # 市场分析
    STOCK_RESEARCH = "stock_research"            # 股票研究
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization" # 组合优化
    RISK_ASSESSMENT = "risk_assessment"          # 风险评估
    STRATEGY_BACKTEST = "strategy_backtest"      # 策略回测
    DATA_QUERY = "data_query"                    # 数据查询
    REPORT_GENERATION = "report_generation"      # 报告生成
    NEWS_ANALYSIS = "news_analysis"              # 新闻分析
    FACTOR_ANALYSIS = "factor_analysis"          # 因子分析
    GENERAL_QUESTION = "general_question"        # 一般问题
    UNKNOWN = "unknown"                          # 未知意图

class ActionType(Enum):
    """动作类型"""
    QUERY_DATA = "query_data"                    # 查询数据
    ANALYZE_DATA = "analyze_data"                # 分析数据
    CALCULATE_METRICS = "calculate_metrics"      # 计算指标
    GENERATE_CHART = "generate_chart"            # 生成图表
    WRITE_REPORT = "write_report"                # 撰写报告
    EXECUTE_STRATEGY = "execute_strategy"        # 执行策略
    OPTIMIZE_PORTFOLIO = "optimize_portfolio"    # 优化组合
    ASSESS_RISK = "assess_risk"                  # 评估风险
    SEARCH_NEWS = "search_news"                  # 搜索新闻
    EXTRACT_FACTORS = "extract_factors"          # 提取因子
    RESPOND = "respond"                          # 回复

@dataclass
class Intent:
    """意图对象"""
    intent_type: IntentType
    confidence: float
    entities: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # 确保置信度在0-1之间
        self.confidence = max(0.0, min(1.0, self.confidence))

@dataclass
class PlanStep:
    """计划步骤"""
    step_id: str
    action: ActionType
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # 依赖的步骤ID
    tools: List[str] = field(default_factory=list)         # 需要的工具
    expected_output: str = ""
    timeout: int = 60  # 超时时间（秒）
    retry_count: int = 3
    
    # 执行状态
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

@dataclass
class ExecutionPlan:
    """执行计划"""
    plan_id: str
    intent: Intent
    steps: List[PlanStep]
    created_at: datetime = field(default_factory=datetime.now)
    estimated_duration: int = 0  # 预估执行时间（秒）
    priority: int = 1  # 优先级
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_next_steps(self) -> List[PlanStep]:
        """获取下一步可执行的步骤"""
        next_steps = []
        
        for step in self.steps:
            if step.status != "pending":
                continue
            
            # 检查依赖是否完成
            dependencies_met = True
            for dep_id in step.dependencies:
                dep_step = next((s for s in self.steps if s.step_id == dep_id), None)
                if not dep_step or dep_step.status != "completed":
                    dependencies_met = False
                    break
            
            if dependencies_met:
                next_steps.append(step)
        
        return next_steps
    
    def is_completed(self) -> bool:
        """检查计划是否完成"""
        return all(step.status in ["completed", "skipped"] for step in self.steps)
    
    def has_failed(self) -> bool:
        """检查计划是否失败"""
        return any(step.status == "failed" for step in self.steps)

class IntentParser:
    """意图解析器"""
    
    def __init__(self):
        self.logger = logging.getLogger("IntentParser")
        
        # 意图识别规则
        self.intent_patterns = {
            IntentType.MARKET_ANALYSIS: [
                r"市场.*分析", r"大盘.*走势", r"市场.*趋势", r"行情.*分析",
                r"market.*analysis", r"market.*trend", r"market.*overview"
            ],
            IntentType.STOCK_RESEARCH: [
                r"股票.*分析", r"个股.*研究", r".*股.*怎么样", r".*股.*分析",
                r"stock.*analysis", r"stock.*research", r"analyze.*stock"
            ],
            IntentType.PORTFOLIO_OPTIMIZATION: [
                r"组合.*优化", r"投资组合", r"资产配置", r"仓位.*优化",
                r"portfolio.*optimization", r"asset.*allocation"
            ],
            IntentType.RISK_ASSESSMENT: [
                r"风险.*评估", r"风险.*分析", r"风险.*控制", r"风险.*管理",
                r"risk.*assessment", r"risk.*analysis", r"risk.*management"
            ],
            IntentType.STRATEGY_BACKTEST: [
                r"策略.*回测", r"回测.*策略", r"策略.*测试", r"历史.*回测",
                r"strategy.*backtest", r"backtest.*strategy"
            ],
            IntentType.DATA_QUERY: [
                r"查询.*数据", r"获取.*数据", r"数据.*查询", r".*数据.*是多少",
                r"query.*data", r"get.*data", r"data.*query"
            ],
            IntentType.REPORT_GENERATION: [
                r"生成.*报告", r"撰写.*报告", r"报告.*生成", r"写.*报告",
                r"generate.*report", r"write.*report", r"create.*report"
            ],
            IntentType.NEWS_ANALYSIS: [
                r"新闻.*分析", r"舆情.*分析", r"消息.*分析", r"资讯.*分析",
                r"news.*analysis", r"sentiment.*analysis"
            ],
            IntentType.FACTOR_ANALYSIS: [
                r"因子.*分析", r"因子.*研究", r"量化.*因子", r"alpha.*因子",
                r"factor.*analysis", r"factor.*research"
            ],
        }
        
        # 实体识别规则
        self.entity_patterns = {
            "stock_code": r"\b\d{6}\b|\b[A-Z]{2,4}\b",  # 股票代码
            "date": r"\d{4}[-/]\d{1,2}[-/]\d{1,2}",      # 日期
            "number": r"\b\d+(?:\.\d+)?\b",              # 数字
            "percentage": r"\b\d+(?:\.\d+)?%\b",         # 百分比
            "money": r"\b\d+(?:\.\d+)?[万亿千百]?元?\b",   # 金额
        }
        
        # 参数提取规则
        self.parameter_extractors = {
            "time_range": self._extract_time_range,
            "stock_symbols": self._extract_stock_symbols,
            "metrics": self._extract_metrics,
            "conditions": self._extract_conditions,
        }
    
    async def parse_intent(self, text: str, context: Optional[Dict[str, Any]] = None) -> Intent:
        """解析用户意图"""
        try:
            # 预处理文本
            processed_text = self._preprocess_text(text)
            
            # 识别意图类型
            intent_type, confidence = self._classify_intent(processed_text)
            
            # 提取实体
            entities = self._extract_entities(processed_text)
            
            # 提取参数
            parameters = self._extract_parameters(processed_text, intent_type)
            
            # 创建意图对象
            intent = Intent(
                intent_type=intent_type,
                confidence=confidence,
                entities=entities,
                parameters=parameters,
                context=context or {}
            )
            
            self.logger.debug(f"Parsed intent: {intent_type.value} (confidence: {confidence:.2f})")
            return intent
            
        except Exception as e:
            self.logger.error(f"Failed to parse intent: {e}")
            return Intent(
                intent_type=IntentType.UNKNOWN,
                confidence=0.0,
                entities={},
                parameters={},
                context=context or {}
            )
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本"""
        # 转换为小写
        text = text.lower()
        
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 标准化标点符号
        text = re.sub(r'[，。！？；：]', ',', text)
        
        return text
    
    def _classify_intent(self, text: str) -> Tuple[IntentType, float]:
        """分类意图"""
        best_intent = IntentType.GENERAL_QUESTION
        best_score = 0.0
        
        for intent_type, patterns in self.intent_patterns.items():
            score = 0.0
            matches = 0
            
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    matches += 1
                    score += 1.0 / len(patterns)  # 权重分配
            
            # 调整分数
            if matches > 0:
                score = score * (1 + matches * 0.1)  # 多匹配加分
                
                if score > best_score:
                    best_score = score
                    best_intent = intent_type
        
        # 标准化置信度
        confidence = min(1.0, best_score)
        
        return best_intent, confidence
    
    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """提取实体"""
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities[entity_type] = matches
        
        return entities
    
    def _extract_parameters(self, text: str, intent_type: IntentType) -> Dict[str, Any]:
        """提取参数"""
        parameters = {}
        
        for param_name, extractor in self.parameter_extractors.items():
            try:
                value = extractor(text, intent_type)
                if value:
                    parameters[param_name] = value
            except Exception as e:
                self.logger.warning(f"Failed to extract parameter {param_name}: {e}")
        
        return parameters
    
    def _extract_time_range(self, text: str, intent_type: IntentType) -> Optional[Dict[str, str]]:
        """提取时间范围"""
        # 查找日期模式
        date_pattern = r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})'
        dates = re.findall(date_pattern, text)
        
        if len(dates) >= 2:
            return {"start_date": dates[0], "end_date": dates[1]}
        elif len(dates) == 1:
            return {"date": dates[0]}
        
        # 查找相对时间
        relative_patterns = {
            r'最近.*?(\d+).*?天': lambda m: {"days": int(m.group(1))},
            r'最近.*?(\d+).*?月': lambda m: {"months": int(m.group(1))},
            r'最近.*?(\d+).*?年': lambda m: {"years": int(m.group(1))},
            r'今天': lambda m: {"days": 1},
            r'本周': lambda m: {"weeks": 1},
            r'本月': lambda m: {"months": 1},
            r'今年': lambda m: {"years": 1},
        }
        
        for pattern, extractor in relative_patterns.items():
            match = re.search(pattern, text)
            if match:
                return extractor(match)
        
        return None
    
    def _extract_stock_symbols(self, text: str, intent_type: IntentType) -> Optional[List[str]]:
        """提取股票代码"""
        # 6位数字代码
        codes = re.findall(r'\b\d{6}\b', text)
        
        # 股票名称（简单匹配）
        stock_names = re.findall(r'([\u4e00-\u9fa5]+)股份?', text)
        
        symbols = codes + stock_names
        return symbols if symbols else None
    
    def _extract_metrics(self, text: str, intent_type: IntentType) -> Optional[List[str]]:
        """提取指标"""
        metric_patterns = {
            r'收益率?': 'return',
            r'波动率?': 'volatility',
            r'夏普比率': 'sharpe_ratio',
            r'最大回撤': 'max_drawdown',
            r'市盈率': 'pe_ratio',
            r'市净率': 'pb_ratio',
            r'ROE': 'roe',
            r'ROA': 'roa',
        }
        
        metrics = []
        for pattern, metric in metric_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                metrics.append(metric)
        
        return metrics if metrics else None
    
    def _extract_conditions(self, text: str, intent_type: IntentType) -> Optional[Dict[str, Any]]:
        """提取条件"""
        conditions = {}
        
        # 数值条件
        value_patterns = {
            r'大于\s*(\d+(?:\.\d+)?)': lambda m: {'operator': '>', 'value': float(m.group(1))},
            r'小于\s*(\d+(?:\.\d+)?)': lambda m: {'operator': '<', 'value': float(m.group(1))},
            r'等于\s*(\d+(?:\.\d+)?)': lambda m: {'operator': '=', 'value': float(m.group(1))},
            r'超过\s*(\d+(?:\.\d+)?)%': lambda m: {'operator': '>', 'value': float(m.group(1)), 'unit': '%'},
        }
        
        for pattern, extractor in value_patterns.items():
            match = re.search(pattern, text)
            if match:
                conditions.update(extractor(match))
                break
        
        return conditions if conditions else None

class TaskPlanner:
    """任务规划器"""
    
    def __init__(self):
        self.logger = logging.getLogger("TaskPlanner")
        self.intent_parser = IntentParser()
        
        # 规划模板
        self.plan_templates = {
            IntentType.MARKET_ANALYSIS: self._create_market_analysis_plan,
            IntentType.STOCK_RESEARCH: self._create_stock_research_plan,
            IntentType.PORTFOLIO_OPTIMIZATION: self._create_portfolio_optimization_plan,
            IntentType.RISK_ASSESSMENT: self._create_risk_assessment_plan,
            IntentType.STRATEGY_BACKTEST: self._create_strategy_backtest_plan,
            IntentType.DATA_QUERY: self._create_data_query_plan,
            IntentType.REPORT_GENERATION: self._create_report_generation_plan,
            IntentType.NEWS_ANALYSIS: self._create_news_analysis_plan,
            IntentType.FACTOR_ANALYSIS: self._create_factor_analysis_plan,
            IntentType.GENERAL_QUESTION: self._create_general_response_plan,
        }
    
    async def create_plan(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> ExecutionPlan:
        """创建执行计划"""
        try:
            # 解析意图
            intent = await self.intent_parser.parse_intent(user_input, context)
            
            # 根据意图类型创建计划
            planner_func = self.plan_templates.get(
                intent.intent_type, 
                self._create_general_response_plan
            )
            
            steps = planner_func(intent, user_input)
            
            # 创建执行计划
            plan = ExecutionPlan(
                plan_id=f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                intent=intent,
                steps=steps,
                estimated_duration=sum(step.timeout for step in steps),
                priority=self._calculate_priority(intent)
            )
            
            self.logger.info(f"Created plan with {len(steps)} steps for intent: {intent.intent_type.value}")
            return plan
            
        except Exception as e:
            self.logger.error(f"Failed to create plan: {e}")
            # 返回默认计划
            return self._create_fallback_plan(user_input)
    
    def _create_market_analysis_plan(self, intent: Intent, user_input: str) -> List[PlanStep]:
        """创建市场分析计划"""
        steps = []
        
        # 1. 获取市场数据
        steps.append(PlanStep(
            step_id="market_data",
            action=ActionType.QUERY_DATA,
            description="获取市场指数和行情数据",
            parameters={
                "data_type": "market_index",
                "symbols": ["000001.SH", "399001.SZ", "399006.SZ"],
                "time_range": intent.parameters.get("time_range", {"days": 30})
            },
            tools=["market_data_connector"],
            expected_output="市场指数历史数据"
        ))
        
        # 2. 分析市场趋势
        steps.append(PlanStep(
            step_id="trend_analysis",
            action=ActionType.ANALYZE_DATA,
            description="分析市场趋势和技术指标",
            parameters={
                "analysis_type": "trend",
                "indicators": ["ma", "rsi", "macd"]
            },
            dependencies=["market_data"],
            tools=["technical_analyzer"],
            expected_output="市场趋势分析结果"
        ))
        
        # 3. 生成分析报告
        steps.append(PlanStep(
            step_id="generate_report",
            action=ActionType.WRITE_REPORT,
            description="生成市场分析报告",
            parameters={
                "report_type": "market_analysis",
                "include_charts": True
            },
            dependencies=["trend_analysis"],
            tools=["report_generator"],
            expected_output="市场分析报告"
        ))
        
        return steps
    
    def _create_stock_research_plan(self, intent: Intent, user_input: str) -> List[PlanStep]:
        """创建股票研究计划"""
        steps = []
        
        symbols = intent.parameters.get("stock_symbols", [])
        if not symbols:
            # 如果没有指定股票，返回错误步骤
            steps.append(PlanStep(
                step_id="error_response",
                action=ActionType.RESPOND,
                description="请指定要分析的股票代码",
                parameters={"message": "请提供具体的股票代码或名称"},
                expected_output="错误提示"
            ))
            return steps
        
        # 1. 获取股票基本信息
        steps.append(PlanStep(
            step_id="stock_info",
            action=ActionType.QUERY_DATA,
            description="获取股票基本信息",
            parameters={
                "data_type": "stock_info",
                "symbols": symbols
            },
            tools=["stock_data_connector"],
            expected_output="股票基本信息"
        ))
        
        # 2. 获取价格数据
        steps.append(PlanStep(
            step_id="price_data",
            action=ActionType.QUERY_DATA,
            description="获取股票价格数据",
            parameters={
                "data_type": "stock_price",
                "symbols": symbols,
                "time_range": intent.parameters.get("time_range", {"days": 90})
            },
            tools=["stock_data_connector"],
            expected_output="股票价格历史数据"
        ))
        
        # 3. 获取财务数据
        steps.append(PlanStep(
            step_id="financial_data",
            action=ActionType.QUERY_DATA,
            description="获取财务数据",
            parameters={
                "data_type": "financial_report",
                "symbols": symbols,
                "report_types": ["income", "balance_sheet"]
            },
            tools=["financial_data_connector"],
            expected_output="财务报表数据"
        ))
        
        # 4. 技术分析
        steps.append(PlanStep(
            step_id="technical_analysis",
            action=ActionType.ANALYZE_DATA,
            description="进行技术分析",
            parameters={
                "analysis_type": "technical",
                "indicators": ["ma", "rsi", "macd", "bollinger"]
            },
            dependencies=["price_data"],
            tools=["technical_analyzer"],
            expected_output="技术分析结果"
        ))
        
        # 5. 基本面分析
        steps.append(PlanStep(
            step_id="fundamental_analysis",
            action=ActionType.ANALYZE_DATA,
            description="进行基本面分析",
            parameters={
                "analysis_type": "fundamental",
                "metrics": ["pe", "pb", "roe", "debt_ratio"]
            },
            dependencies=["financial_data"],
            tools=["fundamental_analyzer"],
            expected_output="基本面分析结果"
        ))
        
        # 6. 生成研究报告
        steps.append(PlanStep(
            step_id="research_report",
            action=ActionType.WRITE_REPORT,
            description="生成股票研究报告",
            parameters={
                "report_type": "stock_research",
                "symbols": symbols,
                "include_recommendation": True
            },
            dependencies=["stock_info", "technical_analysis", "fundamental_analysis"],
            tools=["report_generator"],
            expected_output="股票研究报告"
        ))
        
        return steps
    
    def _create_data_query_plan(self, intent: Intent, user_input: str) -> List[PlanStep]:
        """创建数据查询计划"""
        steps = []
        
        # 简单的数据查询
        steps.append(PlanStep(
            step_id="data_query",
            action=ActionType.QUERY_DATA,
            description="查询请求的数据",
            parameters={
                "query": user_input,
                "entities": intent.entities,
                "time_range": intent.parameters.get("time_range")
            },
            tools=["data_connector"],
            expected_output="查询结果数据"
        ))
        
        # 格式化响应
        steps.append(PlanStep(
            step_id="format_response",
            action=ActionType.RESPOND,
            description="格式化查询结果",
            parameters={
                "format": "table",
                "include_summary": True
            },
            dependencies=["data_query"],
            tools=["formatter"],
            expected_output="格式化的查询结果"
        ))
        
        return steps
    
    def _create_general_response_plan(self, intent: Intent, user_input: str) -> List[PlanStep]:
        """创建一般回复计划"""
        steps = []
        
        steps.append(PlanStep(
            step_id="general_response",
            action=ActionType.RESPOND,
            description="生成一般性回复",
            parameters={
                "user_input": user_input,
                "intent": intent.intent_type.value,
                "confidence": intent.confidence
            },
            tools=["llm_interface"],
            expected_output="回复内容"
        ))
        
        return steps
    
    def _create_portfolio_optimization_plan(self, intent: Intent, user_input: str) -> List[PlanStep]:
        """创建组合优化计划"""
        # 实现组合优化计划逻辑
        return [PlanStep(
            step_id="portfolio_opt",
            action=ActionType.OPTIMIZE_PORTFOLIO,
            description="执行投资组合优化",
            parameters=intent.parameters,
            tools=["portfolio_optimizer"],
            expected_output="优化后的投资组合"
        )]
    
    def _create_risk_assessment_plan(self, intent: Intent, user_input: str) -> List[PlanStep]:
        """创建风险评估计划"""
        # 实现风险评估计划逻辑
        return [PlanStep(
            step_id="risk_assess",
            action=ActionType.ASSESS_RISK,
            description="执行风险评估",
            parameters=intent.parameters,
            tools=["risk_analyzer"],
            expected_output="风险评估报告"
        )]
    
    def _create_strategy_backtest_plan(self, intent: Intent, user_input: str) -> List[PlanStep]:
        """创建策略回测计划"""
        # 实现策略回测计划逻辑
        return [PlanStep(
            step_id="strategy_backtest",
            action=ActionType.EXECUTE_STRATEGY,
            description="执行策略回测",
            parameters=intent.parameters,
            tools=["backtest_engine"],
            expected_output="回测结果"
        )]
    
    def _create_report_generation_plan(self, intent: Intent, user_input: str) -> List[PlanStep]:
        """创建报告生成计划"""
        # 实现报告生成计划逻辑
        return [PlanStep(
            step_id="generate_report",
            action=ActionType.WRITE_REPORT,
            description="生成报告",
            parameters=intent.parameters,
            tools=["report_generator"],
            expected_output="生成的报告"
        )]
    
    def _create_news_analysis_plan(self, intent: Intent, user_input: str) -> List[PlanStep]:
        """创建新闻分析计划"""
        # 实现新闻分析计划逻辑
        return [PlanStep(
            step_id="news_analysis",
            action=ActionType.SEARCH_NEWS,
            description="分析相关新闻",
            parameters=intent.parameters,
            tools=["news_analyzer"],
            expected_output="新闻分析结果"
        )]
    
    def _create_factor_analysis_plan(self, intent: Intent, user_input: str) -> List[PlanStep]:
        """创建因子分析计划"""
        # 实现因子分析计划逻辑
        return [PlanStep(
            step_id="factor_analysis",
            action=ActionType.EXTRACT_FACTORS,
            description="执行因子分析",
            parameters=intent.parameters,
            tools=["factor_analyzer"],
            expected_output="因子分析结果"
        )]
    
    def _calculate_priority(self, intent: Intent) -> int:
        """计算任务优先级"""
        # 根据意图类型和置信度计算优先级
        base_priority = {
            IntentType.MARKET_ANALYSIS: 3,
            IntentType.STOCK_RESEARCH: 4,
            IntentType.PORTFOLIO_OPTIMIZATION: 5,
            IntentType.RISK_ASSESSMENT: 5,
            IntentType.STRATEGY_BACKTEST: 3,
            IntentType.DATA_QUERY: 2,
            IntentType.REPORT_GENERATION: 3,
            IntentType.NEWS_ANALYSIS: 2,
            IntentType.FACTOR_ANALYSIS: 4,
            IntentType.GENERAL_QUESTION: 1,
        }.get(intent.intent_type, 1)
        
        # 根据置信度调整
        confidence_factor = intent.confidence
        
        return max(1, int(base_priority * confidence_factor))
    
    def _create_fallback_plan(self, user_input: str) -> ExecutionPlan:
        """创建后备计划"""
        steps = [PlanStep(
            step_id="fallback_response",
            action=ActionType.RESPOND,
            description="生成后备回复",
            parameters={
                "user_input": user_input,
                "message": "抱歉，我无法理解您的请求。请尝试更具体地描述您的需求。"
            },
            tools=["llm_interface"],
            expected_output="后备回复"
        )]
        
        return ExecutionPlan(
            plan_id=f"fallback_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            intent=Intent(
                intent_type=IntentType.UNKNOWN,
                confidence=0.0
            ),
            steps=steps,
            priority=1
        )

# 主规划器类
class Planner:
    """主规划器"""
    
    def __init__(self):
        self.logger = logging.getLogger("Planner")
        self.task_planner = TaskPlanner()
        self.intent_parser = IntentParser()
    
    async def create_plan(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> ExecutionPlan:
        """创建执行计划"""
        return await self.task_planner.create_plan(user_input, context)
    
    async def parse_intent(self, text: str, context: Optional[Dict[str, Any]] = None) -> Intent:
        """解析意图"""
        return await self.intent_parser.parse_intent(text, context)

# 便捷函数
def create_planner() -> Planner:
    """创建规划器"""
    return Planner()

def create_plan_step(
    step_id: str,
    action: ActionType,
    description: str,
    **kwargs
) -> PlanStep:
    """创建计划步骤"""
    return PlanStep(
        step_id=step_id,
        action=action,
        description=description,
        **kwargs
    )