// @ts-nocheck
// 选股推荐页面
// 实现基于AI和量化分析的智能选股推荐功能

import React, { useState, useEffect } from 'react';
import { clsx } from 'clsx';
import {
  TrendingUp,
  TrendingDown,
  Star,
  Filter,
  Search,
  RefreshCw,
  Eye,
  Heart,
  BarChart3,
  PieChart,
  Target,
  Zap,
  Brain,
  AlertTriangle,
  CheckCircle,
  Clock,
  ArrowUpRight,
  ArrowDownRight,
  Minus,
  Info,
  Download,
  Settings,
} from 'lucide-react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  BarChart,
  Bar,
} from 'recharts';
import { useRecommendations, useRecommendationActions } from '../../store';
import apiService from '../../services/api';
import { format } from 'date-fns';
import { zhCN } from 'date-fns/locale';

// 推荐等级
const RECOMMENDATION_LEVELS = {
  STRONG_BUY: 'strong_buy',
  BUY: 'buy',
  HOLD: 'hold',
  SELL: 'sell',
  STRONG_SELL: 'strong_sell',
} as const;

// 推荐来源
const RECOMMENDATION_SOURCES = {
  AI_MODEL: 'ai_model',
  QUANTITATIVE: 'quantitative',
  FUNDAMENTAL: 'fundamental',
  TECHNICAL: 'technical',
  HYBRID: 'hybrid',
} as const;

// 行业分类
const INDUSTRIES = [
  '全部',
  '科技',
  '金融',
  '医药',
  '消费',
  '制造',
  '能源',
  '房地产',
  '公用事业',
  '材料',
  '通信',
];

// 图表颜色
const CHART_COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

interface Stock {
  symbol: string;
  name: string;
  industry: string;
  price: number;
  change: number;
  change_percent: number;
  volume: number;
  market_cap: number;
  pe_ratio: number;
  pb_ratio: number;
  recommendation: {
    level: string;
    score: number;
    source: string;
    confidence: number;
    target_price: number;
    reasons: string[];
    risk_factors: string[];
  };
  factors: {
    momentum: number;
    value: number;
    quality: number;
    growth: number;
    volatility: number;
    liquidity: number;
  };
  ai_analysis: {
    sentiment_score: number;
    news_impact: number;
    technical_signal: number;
    fundamental_strength: number;
  };
}

interface RecommendationsPageProps {
  className?: string;
}

export const RecommendationsPage: React.FC<RecommendationsPageProps> = ({ className }) => {
  const { recommendations, loading, filters } = useRecommendations();
  const { loadRecommendations, updateFilters, addToWatchlist } = useRecommendationActions();
  
  const [selectedStock, setSelectedStock] = useState<Stock | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [showFilters, setShowFilters] = useState(false);
  const [activeTab, setActiveTab] = useState<'overview' | 'analysis' | 'factors' | 'news'>('overview');
  const [sortBy, setSortBy] = useState<'score' | 'change' | 'volume' | 'market_cap'>('score');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  // 模拟推荐数据
  const [mockRecommendations, setMockRecommendations] = useState<Stock[]>([]);

  // 加载推荐数据
  useEffect(() => {
    loadRecommendations();
    generateMockData();
  }, []);

  // 生成模拟数据
  const generateMockData = () => {
    const mockStocks: Stock[] = [
      {
        symbol: '000001.SZ',
        name: '平安银行',
        industry: '金融',
        price: 12.45,
        change: 0.23,
        change_percent: 1.88,
        volume: 45678900,
        market_cap: 241200000000,
        pe_ratio: 5.2,
        pb_ratio: 0.8,
        recommendation: {
          level: RECOMMENDATION_LEVELS.BUY,
          score: 8.5,
          source: RECOMMENDATION_SOURCES.HYBRID,
          confidence: 0.85,
          target_price: 14.50,
          reasons: ['估值偏低', '业绩稳定增长', '分红收益率高'],
          risk_factors: ['利率风险', '信贷风险']
        },
        factors: {
          momentum: 0.7,
          value: 0.9,
          quality: 0.8,
          growth: 0.6,
          volatility: 0.4,
          liquidity: 0.9
        },
        ai_analysis: {
          sentiment_score: 0.75,
          news_impact: 0.6,
          technical_signal: 0.8,
          fundamental_strength: 0.85
        }
      },
      {
        symbol: '000858.SZ',
        name: '五粮液',
        industry: '消费',
        price: 168.50,
        change: -2.30,
        change_percent: -1.35,
        volume: 12345600,
        market_cap: 652100000000,
        pe_ratio: 28.5,
        pb_ratio: 5.2,
        recommendation: {
          level: RECOMMENDATION_LEVELS.HOLD,
          score: 7.2,
          source: RECOMMENDATION_SOURCES.AI_MODEL,
          confidence: 0.72,
          target_price: 175.00,
          reasons: ['品牌价值高', '消费升级受益', '渠道优势明显'],
          risk_factors: ['估值偏高', '消费放缓风险']
        },
        factors: {
          momentum: 0.5,
          value: 0.3,
          quality: 0.9,
          growth: 0.7,
          volatility: 0.6,
          liquidity: 0.8
        },
        ai_analysis: {
          sentiment_score: 0.65,
          news_impact: 0.7,
          technical_signal: 0.5,
          fundamental_strength: 0.8
        }
      },
      {
        symbol: '300750.SZ',
        name: '宁德时代',
        industry: '科技',
        price: 195.80,
        change: 8.90,
        change_percent: 4.76,
        volume: 23456700,
        market_cap: 860500000000,
        pe_ratio: 45.2,
        pb_ratio: 8.1,
        recommendation: {
          level: RECOMMENDATION_LEVELS.STRONG_BUY,
          score: 9.1,
          source: RECOMMENDATION_SOURCES.QUANTITATIVE,
          confidence: 0.91,
          target_price: 230.00,
          reasons: ['新能源龙头', '技术领先', '订单饱满', '政策支持'],
          risk_factors: ['估值较高', '竞争加剧', '原材料价格波动']
        },
        factors: {
          momentum: 0.9,
          value: 0.2,
          quality: 0.9,
          growth: 0.95,
          volatility: 0.8,
          liquidity: 0.9
        },
        ai_analysis: {
          sentiment_score: 0.9,
          news_impact: 0.85,
          technical_signal: 0.9,
          fundamental_strength: 0.9
        }
      },
      {
        symbol: '600519.SH',
        name: '贵州茅台',
        industry: '消费',
        price: 1680.00,
        change: -15.50,
        change_percent: -0.91,
        volume: 1234560,
        market_cap: 2110000000000,
        pe_ratio: 32.1,
        pb_ratio: 12.5,
        recommendation: {
          level: RECOMMENDATION_LEVELS.HOLD,
          score: 7.8,
          source: RECOMMENDATION_SOURCES.FUNDAMENTAL,
          confidence: 0.78,
          target_price: 1750.00,
          reasons: ['品牌护城河', '盈利能力强', '现金流充沛'],
          risk_factors: ['估值偏高', '增长放缓', '政策风险']
        },
        factors: {
          momentum: 0.4,
          value: 0.2,
          quality: 0.95,
          growth: 0.5,
          volatility: 0.5,
          liquidity: 0.7
        },
        ai_analysis: {
          sentiment_score: 0.7,
          news_impact: 0.6,
          technical_signal: 0.4,
          fundamental_strength: 0.9
        }
      },
      {
        symbol: '002415.SZ',
        name: '海康威视',
        industry: '科技',
        price: 32.15,
        change: 1.05,
        change_percent: 3.37,
        volume: 34567800,
        market_cap: 298700000000,
        pe_ratio: 18.5,
        pb_ratio: 3.2,
        recommendation: {
          level: RECOMMENDATION_LEVELS.BUY,
          score: 8.3,
          source: RECOMMENDATION_SOURCES.TECHNICAL,
          confidence: 0.83,
          target_price: 38.00,
          reasons: ['技术领先', '市场份额高', '估值合理', 'AI应用前景'],
          risk_factors: ['地缘政治风险', '竞争加剧']
        },
        factors: {
          momentum: 0.8,
          value: 0.7,
          quality: 0.8,
          growth: 0.7,
          volatility: 0.6,
          liquidity: 0.8
        },
        ai_analysis: {
          sentiment_score: 0.8,
          news_impact: 0.7,
          technical_signal: 0.85,
          fundamental_strength: 0.8
        }
      }
    ];
    
    setMockRecommendations(mockStocks);
  };

  // 获取推荐等级颜色
  const getRecommendationColor = (level: string) => {
    switch (level) {
      case RECOMMENDATION_LEVELS.STRONG_BUY:
        return 'text-green-700 bg-green-100';
      case RECOMMENDATION_LEVELS.BUY:
        return 'text-green-600 bg-green-50';
      case RECOMMENDATION_LEVELS.HOLD:
        return 'text-yellow-600 bg-yellow-50';
      case RECOMMENDATION_LEVELS.SELL:
        return 'text-red-600 bg-red-50';
      case RECOMMENDATION_LEVELS.STRONG_SELL:
        return 'text-red-700 bg-red-100';
      default:
        return 'text-gray-600 bg-gray-50';
    }
  };

  // 获取推荐等级文本
  const getRecommendationText = (level: string) => {
    switch (level) {
      case RECOMMENDATION_LEVELS.STRONG_BUY:
        return '强烈买入';
      case RECOMMENDATION_LEVELS.BUY:
        return '买入';
      case RECOMMENDATION_LEVELS.HOLD:
        return '持有';
      case RECOMMENDATION_LEVELS.SELL:
        return '卖出';
      case RECOMMENDATION_LEVELS.STRONG_SELL:
        return '强烈卖出';
      default:
        return '未知';
    }
  };

  // 过滤和排序推荐
  const filteredRecommendations = (mockRecommendations || [])
    .filter(stock => {
      if (searchQuery && !stock.name.includes(searchQuery) && !stock.symbol.includes(searchQuery)) {
        return false;
      }
      if (filters.industry && filters.industry !== '全部' && stock.industry !== filters.industry) {
        return false;
      }
      if (filters.recommendation_level && stock.recommendation.level !== filters.recommendation_level) {
        return false;
      }
      return true;
    })
    .sort((a, b) => {
      let aValue, bValue;
      switch (sortBy) {
        case 'score':
          aValue = a.recommendation.score;
          bValue = b.recommendation.score;
          break;
        case 'change':
          aValue = a.change_percent;
          bValue = b.change_percent;
          break;
        case 'volume':
          aValue = a.volume;
          bValue = b.volume;
          break;
        case 'market_cap':
          aValue = a.market_cap;
          bValue = b.market_cap;
          break;
        default:
          return 0;
      }
      
      if (sortOrder === 'desc') {
        return bValue - aValue;
      } else {
        return aValue - bValue;
      }
    });

  // 统计数据
  const stats = {
    total: (mockRecommendations || []).length,
    strong_buy: (mockRecommendations || []).filter(s => s.recommendation.level === RECOMMENDATION_LEVELS.STRONG_BUY).length,
    buy: (mockRecommendations || []).filter(s => s.recommendation.level === RECOMMENDATION_LEVELS.BUY).length,
    hold: (mockRecommendations || []).filter(s => s.recommendation.level === RECOMMENDATION_LEVELS.HOLD).length,
    avg_score: (mockRecommendations || []).length > 0 ? (mockRecommendations || []).reduce((sum, s) => sum + s.recommendation.score, 0) / (mockRecommendations || []).length : 0,
  };

  return (
    <div className={clsx('p-6 space-y-6', className)}>
      {/* 页面标题 */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">智能选股推荐</h1>
          <p className="text-gray-600 mt-1">基于AI和量化分析的个性化选股建议</p>
        </div>
        <div className="flex items-center space-x-3">
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={clsx(
              'btn btn-outline btn-sm',
              showFilters && 'bg-blue-50 text-blue-700 border-blue-300'
            )}
          >
            <Filter className="w-4 h-4 mr-2" />
            筛选
          </button>
          <button
            onClick={() => loadRecommendations()}
            disabled={loading}
            className="btn btn-outline btn-sm"
          >
            <RefreshCw className={clsx('w-4 h-4 mr-2', loading && 'animate-spin')} />
            刷新
          </button>
          <button className="btn btn-primary btn-sm">
            <Download className="w-4 h-4 mr-2" />
            导出报告
          </button>
        </div>
      </div>

      {/* 统计卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
        <div className="card p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">总推荐数</p>
              <p className="text-2xl font-bold text-gray-900">{stats.total}</p>
            </div>
            <div className="p-3 bg-blue-100 rounded-full">
              <Target className="w-6 h-6 text-blue-600" />
            </div>
          </div>
        </div>
        
        <div className="card p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">强烈买入</p>
              <p className="text-2xl font-bold text-green-600">{stats.strong_buy}</p>
            </div>
            <div className="p-3 bg-green-100 rounded-full">
              <TrendingUp className="w-6 h-6 text-green-600" />
            </div>
          </div>
        </div>
        
        <div className="card p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">买入</p>
              <p className="text-2xl font-bold text-blue-600">{stats.buy}</p>
            </div>
            <div className="p-3 bg-blue-100 rounded-full">
              <ArrowUpRight className="w-6 h-6 text-blue-600" />
            </div>
          </div>
        </div>
        
        <div className="card p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">持有</p>
              <p className="text-2xl font-bold text-yellow-600">{stats.hold}</p>
            </div>
            <div className="p-3 bg-yellow-100 rounded-full">
              <Minus className="w-6 h-6 text-yellow-600" />
            </div>
          </div>
        </div>
        
        <div className="card p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">平均评分</p>
              <p className="text-2xl font-bold text-purple-600">{stats.avg_score.toFixed(1)}</p>
            </div>
            <div className="p-3 bg-purple-100 rounded-full">
              <Star className="w-6 h-6 text-purple-600" />
            </div>
          </div>
        </div>
      </div>

      {/* 筛选器 */}
      {showFilters && (
        <div className="card p-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                行业
              </label>
              <select
                value={filters.industry || '全部'}
                onChange={(e) => updateFilters({ industry: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {INDUSTRIES.map(industry => (
                  <option key={industry} value={industry}>{industry}</option>
                ))}
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                推荐等级
              </label>
              <select
                value={filters.recommendation_level || ''}
                onChange={(e) => updateFilters({ recommendation_level: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">全部</option>
                <option value={RECOMMENDATION_LEVELS.STRONG_BUY}>强烈买入</option>
                <option value={RECOMMENDATION_LEVELS.BUY}>买入</option>
                <option value={RECOMMENDATION_LEVELS.HOLD}>持有</option>
                <option value={RECOMMENDATION_LEVELS.SELL}>卖出</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                排序方式
              </label>
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as any)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="score">推荐评分</option>
                <option value="change">涨跌幅</option>
                <option value="volume">成交量</option>
                <option value="market_cap">市值</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                搜索
              </label>
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="搜索股票名称或代码"
                  className="w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 推荐列表 */}
        <div className="lg:col-span-2">
          <div className="card">
            <div className="p-6 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold text-gray-900">推荐股票</h2>
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => setSortOrder(sortOrder === 'desc' ? 'asc' : 'desc')}
                    className="text-sm text-gray-500 hover:text-gray-700"
                  >
                    {sortOrder === 'desc' ? '↓' : '↑'}
                  </button>
                </div>
              </div>
            </div>
            
            <div className="divide-y divide-gray-200 max-h-[600px] overflow-y-auto">
              {filteredRecommendations.map((stock) => (
                <div
                  key={stock.symbol}
                  className={clsx(
                    'p-6 hover:bg-gray-50 cursor-pointer transition-colors',
                    selectedStock?.symbol === stock.symbol && 'bg-blue-50 border-l-4 border-blue-500'
                  )}
                  onClick={() => setSelectedStock(stock)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4">
                      <div>
                        <h3 className="text-sm font-medium text-gray-900">{stock.name}</h3>
                        <p className="text-sm text-gray-500">{stock.symbol} · {stock.industry}</p>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-6">
                      <div className="text-right">
                        <p className="text-sm font-medium text-gray-900">¥{stock.price.toFixed(2)}</p>
                        <p className={clsx(
                          'text-sm',
                          stock.change >= 0 ? 'text-green-600' : 'text-red-600'
                        )}>
                          {stock.change >= 0 ? '+' : ''}{stock.change_percent.toFixed(2)}%
                        </p>
                      </div>
                      
                      <div className="text-right">
                        <div className={clsx(
                          'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium',
                          getRecommendationColor(stock.recommendation.level)
                        )}>
                          {getRecommendationText(stock.recommendation.level)}
                        </div>
                        <p className="text-sm text-gray-500 mt-1">
                          评分: {stock.recommendation.score.toFixed(1)}
                        </p>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            addToWatchlist(stock.symbol);
                          }}
                          className="p-1 text-gray-400 hover:text-red-600"
                        >
                          <Heart className="w-4 h-4" />
                        </button>
                        
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            // 查看详情
                          }}
                          className="p-1 text-gray-400 hover:text-blue-600"
                        >
                          <Eye className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* 股票详情 */}
        <div className="space-y-6">
          {selectedStock ? (
            <>
              {/* 基本信息 */}
              <div className="card p-6">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">
                      {selectedStock.name}
                    </h3>
                    <p className="text-sm text-gray-500">
                      {selectedStock.symbol} · {selectedStock.industry}
                    </p>
                  </div>
                  <div className={clsx(
                    'inline-flex items-center px-3 py-1 rounded-full text-sm font-medium',
                    getRecommendationColor(selectedStock.recommendation.level)
                  )}>
                    {getRecommendationText(selectedStock.recommendation.level)}
                  </div>
                </div>
                
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">当前价格</span>
                    <span className="text-sm font-medium text-gray-900">
                      ¥{selectedStock.price.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">目标价格</span>
                    <span className="text-sm font-medium text-green-600">
                      ¥{selectedStock.recommendation.target_price.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">上涨空间</span>
                    <span className="text-sm font-medium text-green-600">
                      {(((selectedStock.recommendation.target_price - selectedStock.price) / selectedStock.price) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">推荐评分</span>
                    <span className="text-sm font-medium text-gray-900">
                      {selectedStock.recommendation.score.toFixed(1)}/10
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">置信度</span>
                    <span className="text-sm font-medium text-gray-900">
                      {(selectedStock.recommendation.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              </div>

              {/* 推荐理由 */}
              <div className="card p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">推荐理由</h3>
                
                <div className="space-y-4">
                  <div>
                    <h4 className="text-sm font-medium text-green-700 mb-2 flex items-center">
                      <CheckCircle className="w-4 h-4 mr-1" />
                      买入理由
                    </h4>
                    <ul className="space-y-1">
                      {selectedStock.recommendation.reasons.map((reason, index) => (
                        <li key={index} className="text-sm text-gray-600 flex items-start">
                          <span className="w-1.5 h-1.5 bg-green-500 rounded-full mt-2 mr-2 flex-shrink-0" />
                          {reason}
                        </li>
                      ))}
                    </ul>
                  </div>
                  
                  <div>
                    <h4 className="text-sm font-medium text-red-700 mb-2 flex items-center">
                      <AlertTriangle className="w-4 h-4 mr-1" />
                      风险因素
                    </h4>
                    <ul className="space-y-1">
                      {selectedStock.recommendation.risk_factors.map((risk, index) => (
                        <li key={index} className="text-sm text-gray-600 flex items-start">
                          <span className="w-1.5 h-1.5 bg-red-500 rounded-full mt-2 mr-2 flex-shrink-0" />
                          {risk}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>

              {/* 因子分析 */}
              <div className="card p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">因子分析</h3>
                
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart data={[
                      {
                        factor: '动量',
                        value: selectedStock.factors.momentum * 10,
                        fullMark: 10
                      },
                      {
                        factor: '价值',
                        value: selectedStock.factors.value * 10,
                        fullMark: 10
                      },
                      {
                        factor: '质量',
                        value: selectedStock.factors.quality * 10,
                        fullMark: 10
                      },
                      {
                        factor: '成长',
                        value: selectedStock.factors.growth * 10,
                        fullMark: 10
                      },
                      {
                        factor: '波动率',
                        value: (1 - selectedStock.factors.volatility) * 10,
                        fullMark: 10
                      },
                      {
                        factor: '流动性',
                        value: selectedStock.factors.liquidity * 10,
                        fullMark: 10
                      }
                    ]}>
                      <PolarGrid />
                      <PolarAngleAxis dataKey="factor" />
                      <PolarRadiusAxis angle={90} domain={[0, 10]} />
                      <Radar
                        name="因子得分"
                        dataKey="value"
                        stroke="#3b82f6"
                        fill="#3b82f6"
                        fillOpacity={0.3}
                      />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* AI分析 */}
              <div className="card p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                  <Brain className="w-5 h-5 mr-2 text-purple-600" />
                  AI分析
                </h3>
                
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">市场情绪</span>
                    <div className="flex items-center space-x-2">
                      <div className="w-20 bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-blue-600 h-2 rounded-full"
                          style={{ width: `${selectedStock.ai_analysis.sentiment_score * 100}%` }}
                        />
                      </div>
                      <span className="text-sm font-medium text-gray-900 w-8">
                        {(selectedStock.ai_analysis.sentiment_score * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">新闻影响</span>
                    <div className="flex items-center space-x-2">
                      <div className="w-20 bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-green-600 h-2 rounded-full"
                          style={{ width: `${selectedStock.ai_analysis.news_impact * 100}%` }}
                        />
                      </div>
                      <span className="text-sm font-medium text-gray-900 w-8">
                        {(selectedStock.ai_analysis.news_impact * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">技术信号</span>
                    <div className="flex items-center space-x-2">
                      <div className="w-20 bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-purple-600 h-2 rounded-full"
                          style={{ width: `${selectedStock.ai_analysis.technical_signal * 100}%` }}
                        />
                      </div>
                      <span className="text-sm font-medium text-gray-900 w-8">
                        {(selectedStock.ai_analysis.technical_signal * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">基本面强度</span>
                    <div className="flex items-center space-x-2">
                      <div className="w-20 bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-orange-600 h-2 rounded-full"
                          style={{ width: `${selectedStock.ai_analysis.fundamental_strength * 100}%` }}
                        />
                      </div>
                      <span className="text-sm font-medium text-gray-900 w-8">
                        {(selectedStock.ai_analysis.fundamental_strength * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div className="card p-6">
              <div className="text-center py-8">
                <Target className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">选择股票</h3>
                <p className="text-gray-500">点击左侧股票查看详细分析</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default RecommendationsPage;