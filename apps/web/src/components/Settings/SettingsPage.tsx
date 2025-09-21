// @ts-nocheck
import React, { useEffect, useMemo, useState } from 'react';
import { clsx } from 'clsx';
import { ArrowLeft, ArrowRight, CheckCircle2, Sparkles, Star } from 'lucide-react';
import apiService from '../../services/api';
import type { UserPersonalPreferences } from '../../services/api';

const EXPERIENCE_OPTIONS = [
  { id: 'newbie', label: '新手轻松投', description: '希望获得专业陪伴，注重账户安全与稳健收益。' },
  { id: 'experienced', label: '有一定经验', description: '了解市场波动，愿意适度承担风险以追求稳定增值。' },
  { id: 'advanced', label: '经验丰富', description: '熟悉多种策略，关注组合回撤与收益效率。' },
  { id: 'expert', label: '专业资深', description: '擅长使用量化工具，追求行业机会与超额收益。' },
];

const ASSET_OPTIONS = [
  { id: '<100k', label: '10万以内' },
  { id: '100k-1m', label: '10-100万' },
  { id: '>1m', label: '100万以上' },
];

const HORIZON_OPTIONS = [
  '短期 (0-6个月)',
  '中期 (1-3年)',
  '长期 (3年以上)',
];

const STYLE_OPTIONS = [
  { id: 'conservative', label: '稳健型', description: '资产稳步增长，注重风险对冲和本金保护。' },
  { id: 'balanced', label: '平衡型', description: '追求收益与风险平衡，关注资产多样化配置。' },
  { id: 'growth', label: '成长型', description: '聚焦成长行业与主题机会，接受阶段性波动。' },
  { id: 'aggressive', label: '进取型', description: '追求高收益潜力，关注趋势突破与高贝塔资产。' },
];

const INDUSTRY_OPTIONS = [
  '新能源',
  '人工智能',
  '半导体',
  '医药健康',
  '消费升级',
  '高端制造',
  '金融科技',
  '数字经济',
  '绿色能源',
];

const steps = ['投资水平', '风险偏好', '投资风格', '行业偏好', '完成'];

const SettingsPage: React.FC = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [form, setForm] = useState<UserPersonalPreferences | null>(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const loadPreferences = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiService.getUserPreferences();
      if (response.status === 'ok' && response.data) {
        setForm(response.data);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '加载失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadPreferences();
  }, []);

  const updateForm = (updates: Partial<UserPersonalPreferences>) => {
    setForm((prev) => (prev ? { ...prev, ...updates } : prev));
    setSuccess(null);
  };

  const riskAttitudeLabel = useMemo(() => {
    if (!form) return '平衡';
    if (form.risk_score <= 3) return '保守';
    if (form.risk_score <= 6) return '平衡';
    if (form.risk_score <= 8) return '进取';
    return '激进';
  }, [form]);

  const handleRiskScoreChange = (value: number) => {
    const attitude = value <= 3 ? '保守' : value <= 6 ? '平衡' : value <= 8 ? '进取' : '激进';
    updateForm({ risk_score: value, risk_attitude: attitude });
  };

  const toggleIndustry = (industry: string) => {
    if (!form) return;
    const exists = form.focus_industries.includes(industry);
    const next = exists
      ? form.focus_industries.filter((item) => item !== industry)
      : [...form.focus_industries, industry];
    updateForm({ focus_industries: next });
  };

  const handleSubmit = async () => {
    if (!form) return;
    try {
      setSaving(true);
      setSuccess(null);
      const response = await apiService.updateUserPreferences(form);
      if (response.status === 'ok') {
        setSuccess('已保存您的个性化画像，AI 助手将根据最新偏好生成建议。');
        setForm(response.data ?? form);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '保存失败，请稍后再试');
    } finally {
      setSaving(false);
    }
  };

  const handleReset = async () => {
    try {
      setSaving(true);
      const response = await apiService.resetUserPreferences();
      if (response.status === 'ok' && response.data) {
        setForm(response.data);
        setCurrentStep(0);
        setSuccess('已恢复为默认配置，可继续进行个性化定制。');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '重置失败');
    } finally {
      setSaving(false);
    }
  };

  if (loading || !form) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="rounded-3xl border border-white/60 bg-white/80 px-6 py-4 text-slate-500 shadow-soft backdrop-blur-xl">
          <Sparkles className="mr-2 inline h-4 w-4 animate-spin text-sky-500" /> 正在加载个性化配置...
        </div>
      </div>
    );
  }

  const StepContent = () => {
    switch (currentStep) {
      case 0:
        return (
          <div className="grid gap-4 md:grid-cols-2">
            {EXPERIENCE_OPTIONS.map((option) => (
              <button
                key={option.id}
                onClick={() => updateForm({ experience_level: option.id, experience_label: option.label })}
                className={clsx(
                  'rounded-2xl border px-5 py-4 text-left transition-all',
                  form.experience_level === option.id
                    ? 'border-sky-400 bg-sky-50 shadow-soft'
                    : 'border-white/70 bg-white/80 hover:border-sky-200'
                )}
              >
                <div className="text-base font-semibold text-slate-900">{option.label}</div>
                <p className="mt-2 text-sm text-slate-500">{option.description}</p>
              </button>
            ))}
            <div className="rounded-2xl border border-white/60 bg-white/70 p-5 shadow-inner">
              <div className="text-sm font-medium text-slate-500">投资资金规模</div>
              <div className="mt-3 flex flex-wrap gap-2">
                {ASSET_OPTIONS.map((asset) => (
                  <button
                    key={asset.id}
                    onClick={() => updateForm({ asset_scale: asset.id })}
                    className={clsx(
                      'rounded-full px-4 py-2 text-sm font-medium transition-colors',
                      form.asset_scale === asset.id
                        ? 'bg-gradient-to-r from-sky-500 to-blue-500 text-white shadow-soft'
                        : 'bg-slate-100/70 text-slate-600 hover:bg-slate-200'
                    )}
                  >
                    {asset.label}
                  </button>
                ))}
              </div>
            </div>
          </div>
        );
      case 1:
        return (
          <div className="space-y-6">
            <div className="rounded-2xl border border-white/60 bg-white/80 p-6 shadow-inner">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-sm font-medium text-slate-600">风险承受程度</div>
                  <div className="mt-1 text-xl font-semibold text-slate-900">{form.risk_score} / 10</div>
                </div>
                <div className="rounded-full bg-sky-50/80 px-4 py-2 text-sm text-sky-600">倾向：{riskAttitudeLabel}</div>
              </div>
              <input
                type="range"
                min={1}
                max={10}
                value={form.risk_score}
                onChange={(event) => handleRiskScoreChange(Number(event.target.value))}
                className="mt-6 h-2 w-full cursor-pointer appearance-none rounded-full bg-slate-200"
              />
            </div>
            <div className="rounded-2xl border border-white/60 bg-white/80 p-6 shadow-inner">
              <div className="text-sm font-medium text-slate-600">投资周期偏好</div>
              <div className="mt-3 flex flex-wrap gap-3">
                {HORIZON_OPTIONS.map((option) => (
                  <button
                    key={option}
                    onClick={() => updateForm({ investment_horizon: option })}
                    className={clsx(
                      'rounded-full px-4 py-2 text-sm transition-all',
                      form.investment_horizon === option
                        ? 'bg-gradient-to-r from-sky-500 to-blue-500 text-white shadow-soft'
                        : 'bg-slate-100/70 text-slate-600 hover:bg-slate-200'
                    )}
                  >
                    {option}
                  </button>
                ))}
              </div>
            </div>
          </div>
        );
      case 2:
        return (
          <div className="grid gap-4 md:grid-cols-2">
            {STYLE_OPTIONS.map((style) => (
              <button
                key={style.id}
                onClick={() => updateForm({ strategy_style: style.id })}
                className={clsx(
                  'rounded-2xl border px-5 py-4 text-left transition-all',
                  form.strategy_style === style.id
                    ? 'border-blue-400 bg-blue-50 shadow-soft'
                    : 'border-white/70 bg-white/80 hover:border-blue-200'
                )}
              >
                <div className="text-base font-semibold text-slate-900">{style.label}</div>
                <p className="mt-2 text-sm text-slate-500">{style.description}</p>
              </button>
            ))}
          </div>
        );
      case 3:
        return (
          <div className="space-y-4">
            <div className="text-sm font-medium text-slate-600">关注的行业方向</div>
            <div className="flex flex-wrap gap-2">
              {INDUSTRY_OPTIONS.map((industry) => (
                <button
                  key={industry}
                  onClick={() => toggleIndustry(industry)}
                  className={clsx(
                    'rounded-full px-4 py-2 text-sm transition-colors',
                    form.focus_industries.includes(industry)
                      ? 'bg-gradient-to-r from-sky-500 to-blue-500 text-white shadow-soft'
                      : 'bg-slate-100/70 text-slate-600 hover:bg-slate-200'
                  )}
                >
                  {industry}
                </button>
              ))}
            </div>
          </div>
        );
      case 4:
        return (
          <div className="space-y-4">
            <div className="rounded-2xl border border-sky-200/70 bg-sky-50/70 p-6 shadow-soft">
              <div className="flex items-center gap-3 text-sky-700">
                <CheckCircle2 className="h-6 w-6" />
                <div>
                  <div className="text-lg font-semibold">个性化画像已生成</div>
                  <p className="text-sm text-sky-600">AI 助手会基于以下偏好提供策略建议和执行清单。</p>
                </div>
              </div>
            </div>
            <div className="grid gap-4 md:grid-cols-2">
              <SummaryCard label="投资水平" value={form.experience_label} />
              <SummaryCard label="资金规模" value={ASSET_OPTIONS.find((item) => item.id === form.asset_scale)?.label || form.asset_scale} />
              <SummaryCard label="风险偏好" value={`${riskAttitudeLabel} (${form.risk_score}/10)`} />
              <SummaryCard label="投资周期" value={form.investment_horizon} />
              <SummaryCard label="策略风格" value={STYLE_OPTIONS.find((item) => item.id === form.strategy_style)?.label || form.strategy_style} />
              <SummaryCard label="关注行业" value={form.focus_industries.join('、') || '待选择'} />
            </div>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="relative z-10 mx-auto flex h-full max-w-5xl flex-col space-y-6 py-6">
      <div className="rounded-3xl border border-white/60 bg-white/80 p-8 shadow-soft backdrop-blur-xl">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm font-medium text-sky-600">个性化配置向导</div>
            <h1 className="mt-2 text-3xl font-semibold text-slate-900">打造专属的投资画像</h1>
            <p className="mt-2 text-sm text-slate-500">填写以下信息，AI 助手将根据您的投资风格和偏好提供策略建议。</p>
          </div>
          <div className="rounded-2xl bg-sky-50/80 px-4 py-3 text-sm text-sky-600">
            <div className="font-medium">步骤 {currentStep + 1} / {steps.length}</div>
            <div className="text-xs text-slate-500">{steps[currentStep]}</div>
          </div>
        </div>

        <div className="mt-6 flex flex-wrap gap-2">
          {steps.map((step, index) => (
            <div
              key={step}
              className={clsx(
                'flex items-center rounded-full px-4 py-2 text-xs transition-colors',
                index === currentStep
                  ? 'bg-gradient-to-r from-sky-500 to-blue-500 text-white shadow-soft'
                  : index < currentStep
                    ? 'bg-sky-100 text-sky-600'
                    : 'bg-slate-100/70 text-slate-500'
              )}
            >
              {index + 1}. {step}
            </div>
          ))}
        </div>

        <div className="mt-8">
          <StepContent />
        </div>

        <div className="mt-8 flex flex-wrap items-center justify-between gap-3 border-t border-white/60 pt-6">
          <button
            onClick={() => setCurrentStep((prev) => Math.max(prev - 1, 0))}
            disabled={currentStep === 0}
            className={clsx(
              'inline-flex items-center rounded-full px-4 py-2 text-sm transition-colors',
              currentStep === 0
                ? 'cursor-not-allowed bg-slate-100 text-slate-400'
                : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            )}
          >
            <ArrowLeft className="mr-2 h-4 w-4" /> 上一步
          </button>

          <div className="flex items-center gap-3">
            <button
              onClick={handleReset}
              disabled={saving}
              className="rounded-full bg-white/80 px-4 py-2 text-sm text-slate-500 transition-colors hover:bg-white"
            >
              恢复默认
            </button>

            {currentStep < steps.length - 1 ? (
              <button
                onClick={() => setCurrentStep((prev) => Math.min(prev + 1, steps.length - 1))}
                className="inline-flex items-center rounded-full bg-gradient-to-r from-sky-500 to-blue-500 px-5 py-2 text-sm font-medium text-white shadow-soft transition-all hover:translate-x-0.5"
              >
                下一步
                <ArrowRight className="ml-2 h-4 w-4" />
              </button>
            ) : (
              <button
                onClick={handleSubmit}
                disabled={saving}
                className="inline-flex items-center rounded-full bg-gradient-to-r from-blue-500 to-indigo-500 px-6 py-2 text-sm font-medium text-white shadow-soft transition-all"
              >
                <Star className="mr-2 h-4 w-4" /> 保存设置
              </button>
            )}
          </div>
        </div>

        {success && (
          <div className="mt-6 rounded-2xl border border-emerald-200/70 bg-emerald-50/70 px-4 py-3 text-sm text-emerald-600 shadow-soft">
            {success}
          </div>
        )}

        {error && (
          <div className="mt-4 rounded-2xl border border-red-200/70 bg-red-50/70 px-4 py-3 text-sm text-red-600 shadow-soft">
            {error}
          </div>
        )}
      </div>
    </div>
  );
};

interface SummaryCardProps {
  label: string;
  value: string;
}

const SummaryCard: React.FC<SummaryCardProps> = ({ label, value }) => (
  <div className="rounded-2xl border border-white/60 bg-white/80 p-5 shadow-inner">
    <div className="text-xs font-medium uppercase tracking-wider text-slate-400">{label}</div>
    <div className="mt-2 text-lg font-semibold text-slate-900">{value}</div>
  </div>
);

export default SettingsPage;

