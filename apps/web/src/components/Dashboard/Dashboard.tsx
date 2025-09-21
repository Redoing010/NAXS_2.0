// @ts-nocheck
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { clsx } from 'clsx';
import {
  ArrowUpRight,
  Calendar,
  ChevronRight,
  MessageCircle,
  Send,
  ShieldCheck,
  Sparkles,
  Star,
  TrendingUp,
  Zap,
} from 'lucide-react';
import {
  Area,
  AreaChart,
  Bar,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import apiService from '../../services/api';
import type {
  AssistantChatResult,
  DashboardOverview,
  UserPersonalPreferences,
} from '../../services/api';

interface ChatMessageItem {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  insights?: string[];
}

const currencyFormatter = new Intl.NumberFormat('zh-CN', {
  style: 'currency',
  currency: 'CNY',
  maximumFractionDigits: 2,
});

const numberFormatter = new Intl.NumberFormat('zh-CN');

const chartTooltipStyle = {
  backgroundColor: 'rgba(255, 255, 255, 0.95)',
  border: '1px solid rgba(148, 163, 184, 0.2)',
  borderRadius: '12px',
  padding: '12px',
};

const Dashboard: React.FC = () => {
  const [overview, setOverview] = useState<DashboardOverview | null>(null);
  const [briefing, setBriefing] = useState<{ title: string; highlights: string[] } | null>(null);
  const [preferences, setPreferences] = useState<UserPersonalPreferences | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [quickPrompts, setQuickPrompts] = useState<string[]>([]);
  const [messages, setMessages] = useState<ChatMessageItem[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [sending, setSending] = useState(false);

  const chatEndRef = useRef<HTMLDivElement>(null);

  const headline = overview?.headline || 'AI 驱动的下一代智能投研平台';
  const subheadline = overview?.subheadline || 'NAXS 将实时市场数据与智能分析融合，陪伴你完成每一次关键决策。';

  const loadOverview = async () => {
    try {
      setLoading(true);
      setError(null);

      const [dashboardRes, briefingRes, promptsRes, preferencesRes] = await Promise.all([
        apiService.getDashboardOverview(),
        apiService.getDashboardBriefing(),
        apiService.getAssistantPrompts(),
        apiService.getUserPreferences(),
      ]);

      if (dashboardRes.status === 'ok' && dashboardRes.data) {
        setOverview(dashboardRes.data);
        if (messages.length === 0) {
          const greeting = dashboardRes.data.profile?.greeting || `您好，${dashboardRes.data.profile?.name || '投资者'}`;
          setMessages([
            {
              id: 'assistant-initial',
              role: 'assistant',
              content: greeting,
              timestamp: dashboardRes.data.timestamp,
            },
          ]);
        }
      }

      if (briefingRes.status === 'ok' && briefingRes.data) {
        setBriefing(briefingRes.data);
      }

      if (promptsRes.status === 'ok' && promptsRes.data) {
        setQuickPrompts(promptsRes.data.items || []);
      }

      if (preferencesRes.status === 'ok' && preferencesRes.data) {
        setPreferences(preferencesRes.data);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '加载数据失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadOverview();
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async (preset?: string) => {
    const content = (preset ?? inputValue).trim();
    if (!content || sending) return;

    const userMessage: ChatMessageItem = {
      id: `user-${Date.now()}`,
      role: 'user',
      content,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue('');
    setSending(true);

    try {
      const response = await apiService.sendAssistantMessage(content);
      if (response.status === 'ok' && response.data) {
        appendAssistantMessage(response.data);
      } else {
        appendAssistantMessage({
          reply: '抱歉，当前无法获取智能回复，请稍后再试。',
          insights: [],
          timestamp: new Date().toISOString(),
        });
      }
    } catch (err) {
      appendAssistantMessage({
        reply: '网络连接出现波动，请稍后再试。',
        insights: [],
        timestamp: new Date().toISOString(),
      });
    } finally {
      setSending(false);
    }
  };

  const appendAssistantMessage = (result: AssistantChatResult) => {
    const assistantMessage: ChatMessageItem = {
      id: `assistant-${Date.now()}`,
      role: 'assistant',
      content: result.reply,
      timestamp: result.timestamp,
      insights: result.insights,
    };
    setMessages((prev) => [...prev, assistantMessage]);
  };

  const account = overview?.account;
  const marketHeat = overview?.market_heat;
  const performanceData = overview?.performance_trend ?? [];
  const fundFlowData = overview?.fund_flow ?? [];
  const aiInsights = overview?.ai_insights ?? [];
  const newsItems = overview?.news ?? [];
  const profile = overview?.profile;

  const riskTag = useMemo(() => {
    if (!account) return '风险状态：未知';
    if (account.risk_score <= 3) return '风险状态：保守';
    if (account.risk_score <= 6) return '风险状态：平衡';
    if (account.risk_score <= 8) return '风险状态：进取';
    return '风险状态：激进';
  }, [account]);

  return (
    <div className="relative z-10 h-full overflow-hidden">
      <div className="grid h-full grid-cols-1 gap-6 xl:grid-cols-[420px,1fr]">
        <div className="flex h-full flex-col space-y-6">
          <div className="flex flex-col rounded-3xl border border-white/50 bg-white/70 p-6 shadow-soft backdrop-blur-xl">
            <div className="flex items-start justify-between">
              <div>
                <div className="flex items-center space-x-2 text-sm text-sky-600">
                  <Sparkles className="h-4 w-4" />
                  <span>智能投研助手</span>
                </div>
                <h2 className="mt-3 text-2xl font-semibold text-slate-900">
                  {profile?.name ? `${profile.name}，` : ''}欢迎回来
                </h2>
                <p className="mt-2 text-sm text-slate-500">
                  {profile?.greeting || '让我们一起查看今日的市场机会，并快速制定执行计划。'}
                </p>
              </div>
              {profile?.avatar && (
                <img
                  src={profile.avatar}
                  alt={profile.name}
                  className="h-14 w-14 rounded-2xl border border-white/60 object-cover shadow-soft"
                />
              )}
            </div>

            <div className="mt-4 flex flex-wrap gap-2">
              {profile?.badges?.map((badge) => (
                <span
                  key={badge}
                  className="rounded-full bg-sky-50/80 px-3 py-1 text-xs font-medium text-sky-600"
                >
                  {badge}
                </span>
              ))}
              {preferences && (
                <span className="rounded-full bg-blue-50/80 px-3 py-1 text-xs font-medium text-blue-600">
                  {preferences.strategy_style === 'balanced' ? '策略偏好：均衡' : `策略偏好：${preferences.strategy_style}`}
                </span>
              )}
            </div>

            <div className="mt-6 space-y-3">
              <div className="text-xs font-medium uppercase tracking-wide text-slate-400">
                今日建议提问
              </div>
              <div className="grid gap-2">
                {quickPrompts.slice(0, 3).map((prompt) => (
                  <button
                    key={prompt}
                    onClick={() => handleSendMessage(prompt)}
                    className="flex items-center justify-between rounded-2xl border border-transparent bg-slate-50/80 px-4 py-3 text-left text-sm text-slate-600 transition-all hover:-translate-y-0.5 hover:border-sky-200 hover:bg-white/90"
                  >
                    <span className="flex-1">{prompt}</span>
                    <ChevronRight className="h-4 w-4 text-slate-400" />
                  </button>
                ))}
              </div>
            </div>

            <div className="mt-6 flex-1 overflow-hidden">
              <div className="rounded-2xl bg-white/80 p-4 shadow-inner">
                <div className="flex items-center text-sm font-medium text-slate-500">
                  <MessageCircle className="mr-2 h-4 w-4" />
                  智能对话
                </div>
                <div className="mt-3 max-h-64 space-y-4 overflow-y-auto pr-1">
                  {messages.map((message) => (
                    <div key={message.id} className="space-y-1">
                      <div
                        className={clsx(
                          'inline-block rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-soft',
                          message.role === 'assistant'
                            ? 'bg-gradient-to-br from-sky-500/90 to-blue-500/90 text-white'
                            : 'bg-slate-100 text-slate-700'
                        )}
                      >
                        <div className="whitespace-pre-line">{message.content}</div>
                        {message.insights && message.insights.length > 0 && (
                          <ul className="mt-2 space-y-1 text-xs text-white/80">
                            {message.insights.map((item, idx) => (
                              <li key={idx} className="flex items-start">
                                <span className="mr-2 mt-1 inline-block h-1.5 w-1.5 rounded-full bg-white/70" />
                                <span className="flex-1">{item}</span>
                              </li>
                            ))}
                          </ul>
                        )}
                      </div>
                      <div className="text-xs text-slate-400">
                        {new Date(message.timestamp).toLocaleString('zh-CN', { hour: '2-digit', minute: '2-digit' })}
                      </div>
                    </div>
                  ))}
                  <div ref={chatEndRef} />
                </div>
                <div className="mt-4 flex items-center rounded-2xl border border-slate-200/70 bg-white/70 px-3 py-2 shadow-inner">
                  <input
                    type="text"
                    value={inputValue}
                    placeholder="请输入您的需求，例如：帮我评估今日的风险敞口"
                    onChange={(event) => setInputValue(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter') {
                        event.preventDefault();
                        handleSendMessage();
                      }
                    }}
                    className="flex-1 bg-transparent text-sm text-slate-600 placeholder:text-slate-400 focus:outline-none"
                  />
                  <button
                    onClick={() => handleSendMessage()}
                    disabled={sending}
                    className={clsx(
                      'ml-2 inline-flex h-10 w-10 items-center justify-center rounded-full bg-gradient-to-br from-sky-500 to-blue-500 text-white shadow-soft transition-all',
                      sending && 'opacity-60'
                    )}
                  >
                    <Send className="h-4 w-4" />
                  </button>
                </div>
              </div>
            </div>
          </div>

          {briefing && (
            <div className="rounded-3xl border border-white/50 bg-white/70 p-6 shadow-soft backdrop-blur-xl">
              <div className="flex items-center space-x-2 text-sm text-slate-500">
                <Calendar className="h-4 w-4" />
                <span>{briefing.title}</span>
              </div>
              <ul className="mt-3 space-y-2 text-sm text-slate-600">
                {briefing.highlights.map((item, idx) => (
                  <li key={idx} className="flex items-start">
                    <span className="mr-2 mt-1 inline-block h-1.5 w-1.5 rounded-full bg-sky-500/70" />
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        <div className="flex h-full flex-col space-y-6">
          <div className="rounded-3xl border border-white/60 bg-white/80 p-6 shadow-soft backdrop-blur-xl">
            <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
              <div>
                <h1 className="text-3xl font-semibold text-slate-900">{headline}</h1>
                <p className="mt-2 text-sm text-slate-500">{subheadline}</p>
              </div>
              {account && (
                <div className="flex items-center gap-4">
                  <div className="rounded-2xl bg-sky-50/80 px-4 py-2 text-sm text-sky-600">
                    <span className="font-medium">综合评分</span>
                    <span className="ml-2 text-lg font-semibold">{account.score}</span>
                  </div>
                  <div className="rounded-2xl bg-blue-50/80 px-4 py-2 text-sm text-blue-600">
                    <span className="font-medium">进度</span>
                    <span className="ml-2 text-lg font-semibold">
                      {account.goal.completed}/{account.goal.target}
                    </span>
                  </div>
                </div>
              )}
            </div>

            <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-3">
              {account && (
                <>
                  <MetricCard
                    title="净资产"
                    value={currencyFormatter.format(account.net_asset)}
                    description={`风险评分 ${account.risk_score}/10`}
                    icon={<ShieldCheck className="h-5 w-5" />}
                    accent="from-sky-500 to-blue-500"
                  />
                  <MetricCard
                    title="累计收益"
                    value={currencyFormatter.format(account.week_pnl)}
                    description={account.risk_comment}
                    icon={<TrendingUp className="h-5 w-5" />}
                    accent="from-emerald-400 to-teal-500"
                  />
                  <MetricCard
                    title="可用资金"
                    value={currencyFormatter.format(account.available_cash)}
                    description={riskTag}
                    icon={<Star className="h-5 w-5" />}
                    accent="from-amber-400 to-orange-500"
                  />
                </>
              )}
            </div>
          </div>

          <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
            <div className="rounded-3xl border border-white/60 bg-white/80 p-6 shadow-soft backdrop-blur-xl">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-semibold text-slate-900">资金净流与AI预测</h2>
                  <p className="text-sm text-slate-500">结合北向资金与量化模型的流向评估</p>
                </div>
                {marketHeat && (
                  <div className="rounded-xl bg-sky-50/80 px-3 py-2 text-right text-xs text-sky-600">
                    <div>市场热度 {marketHeat.score.toFixed(1)}</div>
                    <div className="text-slate-400">AI 情绪：{marketHeat.ai_sentiment}</div>
                  </div>
                )}
              </div>
              <div className="mt-6 h-64">
                <ResponsiveContainer>
                  <ComposedChart data={fundFlowData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="date" stroke="#94a3b8" />
                    <YAxis
                      stroke="#94a3b8"
                      tickFormatter={(value) => `${(value / 1000).toFixed(1)}k`}
                    />
                    <Tooltip
                      contentStyle={chartTooltipStyle}
                      formatter={(value: number, name: string) => [
                        `${numberFormatter.format(value)} 万`,
                        name === 'net_flow' ? '净流入' : 'AI 预测',
                      ]}
                    />
                    <Bar dataKey="net_flow" barSize={24} radius={[16, 16, 16, 16]} fill="#38bdf8" />
                    <Line type="monotone" dataKey="forecast" stroke="#22d3ee" strokeWidth={3} dot={false} />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="rounded-3xl border border-white/60 bg-white/80 p-6 shadow-soft backdrop-blur-xl">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-semibold text-slate-900">组合表现对比</h2>
                  <p className="text-sm text-slate-500">组合 VS 基准指数（近 10 日）</p>
                </div>
                <div className="rounded-xl bg-blue-50/80 px-3 py-2 text-xs text-blue-600">
                  <div className="flex items-center gap-2">
                    <span className="flex items-center text-[10px] font-medium uppercase text-slate-500">
                      <span className="mr-1 inline-block h-2 w-2 rounded-full bg-blue-500" />组合
                    </span>
                    <span className="flex items-center text-[10px] font-medium uppercase text-slate-500">
                      <span className="mr-1 inline-block h-2 w-2 rounded-full bg-emerald-400" />基准
                    </span>
                  </div>
                </div>
              </div>
              <div className="mt-6 h-64">
                <ResponsiveContainer>
                  <AreaChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="date" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" />
                    <Tooltip
                      contentStyle={chartTooltipStyle}
                      formatter={(value: number, name: string) => [
                        currencyFormatter.format(value),
                        name === 'portfolio' ? '组合' : '基准',
                      ]}
                    />
                    <Area type="monotone" dataKey="portfolio" stroke="#2563eb" fill="#3b82f6" fillOpacity={0.2} strokeWidth={2} />
                    <Area type="monotone" dataKey="benchmark" stroke="#10b981" fill="#34d399" fillOpacity={0.2} strokeWidth={2} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1.5fr,1fr]">
            <div className="rounded-3xl border border-white/60 bg-white/80 p-6 shadow-soft backdrop-blur-xl">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-semibold text-slate-900">AI 洞察</h2>
                  <p className="text-sm text-slate-500">根据偏好生成的策略建议与执行要点</p>
                </div>
                <button className="inline-flex items-center text-sm text-sky-600 hover:text-sky-500">
                  查看全部
                  <ArrowUpRight className="ml-1 h-4 w-4" />
                </button>
              </div>
              <div className="mt-4 space-y-4">
                {aiInsights.map((insight) => (
                  <div key={insight.title} className="rounded-2xl border border-slate-100/80 bg-white/90 p-4 shadow-inner">
                    <div className="flex items-center gap-3">
                      <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-sky-100 text-sky-600">
                        <Zap className="h-4 w-4" />
                      </div>
                      <div>
                        <div className="text-sm font-semibold text-slate-800">{insight.title}</div>
                        <p className="mt-1 text-sm text-slate-500">{insight.detail}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-3xl border border-white/60 bg-white/80 p-6 shadow-soft backdrop-blur-xl">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold text-slate-900">市场速览</h2>
                <span className="text-xs text-slate-400">实时更新</span>
              </div>
              <ul className="mt-4 space-y-4">
                {newsItems.map((item) => (
                  <li key={item.title} className="rounded-2xl bg-slate-50/70 p-4">
                    <div className="text-sm font-semibold text-slate-800">{item.title}</div>
                    <div className="mt-2 flex items-center justify-between text-xs text-slate-500">
                      <span>{item.source}</span>
                      <span>{item.timestamp}</span>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </div>

      {loading && (
        <div className="absolute inset-0 flex items-center justify-center rounded-3xl bg-white/60 backdrop-blur-md">
          <div className="flex items-center gap-3 text-slate-500">
            <Sparkles className="h-5 w-5 animate-spin text-sky-500" />
            <span>正在加载智能概览...</span>
          </div>
        </div>
      )}

      {error && (
        <div className="absolute bottom-6 right-6 rounded-2xl border border-red-200 bg-white/90 px-4 py-3 text-sm text-red-500 shadow-soft">
          {error}
        </div>
      )}
    </div>
  );
};

interface MetricCardProps {
  title: string;
  value: string;
  description: string;
  icon: React.ReactNode;
  accent: string;
}

const MetricCard: React.FC<MetricCardProps> = ({ title, value, description, icon, accent }) => (
  <div className="flex flex-col rounded-2xl border border-slate-100/80 bg-white/90 p-4 shadow-inner">
    <div className={clsx('mb-3 inline-flex h-11 w-11 items-center justify-center rounded-xl text-white', `bg-gradient-to-br ${accent}`)}>
      {icon}
    </div>
    <div className="text-sm font-medium text-slate-500">{title}</div>
    <div className="mt-1 text-2xl font-semibold text-slate-900">{value}</div>
    <div className="mt-2 text-xs text-slate-400">{description}</div>
  </div>
);

export default Dashboard;

