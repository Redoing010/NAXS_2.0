// @ts-nocheck
import React, { memo, lazy, Suspense, useCallback, useMemo, useRef, useEffect } from 'react';
import { ErrorBoundary } from 'react-error-boundary';
import { usePerformanceMonitor, useMemoryLeak, useRenderOptimization } from '../../hooks/usePerformance';

// 性能监控组件
export const PerformanceMonitor: React.FC<{
  children: React.ReactNode;
  componentName: string;
  enableLogging?: boolean;
}> = ({ children, componentName, enableLogging = false }) => {
  const { startRender, endRender, getMetrics } = usePerformanceMonitor();
  const { trackRender } = useRenderOptimization();
  
  useMemoryLeak(componentName);
  
  useEffect(() => {
    startRender();
    trackRender({}, componentName);
    
    return () => {
      endRender();
      
      if (enableLogging) {
        const metrics = getMetrics();
        console.log(`[Performance] ${componentName}:`, metrics);
      }
    };
  });
  
  return <>{children}</>;
};

// 虚拟滚动列表组件
export const VirtualScrollList: React.FC<{
  items: any[];
  itemHeight: number;
  containerHeight: number;
  renderItem: (item: any, index: number) => React.ReactNode;
  className?: string;
}> = memo(({ items, itemHeight, containerHeight, renderItem, className }) => {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = React.useState(0);
  
  const visibleItems = useMemo(() => {
    const startIndex = Math.floor(scrollTop / itemHeight);
    const endIndex = Math.min(
      items.length - 1,
      Math.ceil((scrollTop + containerHeight) / itemHeight) + 5
    );
    
    return items.slice(Math.max(0, startIndex - 5), endIndex + 1).map((item, index) => ({
      item,
      index: startIndex + index - 5
    }));
  }, [items, itemHeight, containerHeight, scrollTop]);
  
  const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    setScrollTop(e.currentTarget.scrollTop);
  }, []);
  
  const totalHeight = items.length * itemHeight;
  const offsetY = Math.max(0, Math.floor(scrollTop / itemHeight) - 5) * itemHeight;
  
  return (
    <div
      ref={scrollRef}
      className={className}
      style={{ height: containerHeight, overflow: 'auto' }}
      onScroll={handleScroll}
    >
      <div style={{ height: totalHeight, position: 'relative' }}>
        <div style={{ transform: `translateY(${offsetY}px)` }}>
          {visibleItems.map(({ item, index }) => (
            <div key={index} style={{ height: itemHeight }}>
              {renderItem(item, index)}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
});

// 懒加载图片组件
export const LazyImage: React.FC<{
  src: string;
  alt: string;
  placeholder?: string;
  className?: string;
  onLoad?: () => void;
  onError?: () => void;
}> = memo(({ src, alt, placeholder, className, onLoad, onError }) => {
  const [isLoaded, setIsLoaded] = React.useState(false);
  const [isInView, setIsInView] = React.useState(false);
  const imgRef = useRef<HTMLImageElement>(null);
  
  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsInView(true);
          observer.disconnect();
        }
      },
      { threshold: 0.1 }
    );
    
    if (imgRef.current) {
      observer.observe(imgRef.current);
    }
    
    return () => observer.disconnect();
  }, []);
  
  const handleLoad = useCallback(() => {
    setIsLoaded(true);
    onLoad?.();
  }, [onLoad]);
  
  const handleError = useCallback(() => {
    onError?.();
  }, [onError]);
  
  return (
    <div ref={imgRef} className={className}>
      {isInView && (
        <img
          src={isLoaded ? src : placeholder}
          alt={alt}
          onLoad={handleLoad}
          onError={handleError}
          style={{
            opacity: isLoaded ? 1 : 0.5,
            transition: 'opacity 0.3s ease'
          }}
        />
      )}
    </div>
  );
});

// 代码分割组件包装器
export const CodeSplitWrapper: React.FC<{
  loader: () => Promise<{ default: React.ComponentType<any> }>;
  fallback?: React.ReactNode;
  errorFallback?: React.ComponentType<{ error: Error; resetErrorBoundary: () => void }>;
  props?: any;
}> = ({ loader, fallback, errorFallback, props = {} }) => {
  const LazyComponent = useMemo(() => lazy(loader), [loader]);
  
  const DefaultErrorFallback: React.FC<{ error: Error; resetErrorBoundary: () => void }> = 
    ({ error, resetErrorBoundary }) => (
      <div className="p-4 border border-red-300 rounded-lg bg-red-50">
        <h3 className="text-red-800 font-semibold mb-2">组件加载失败</h3>
        <p className="text-red-600 text-sm mb-3">{error.message}</p>
        <button
          onClick={resetErrorBoundary}
          className="px-3 py-1 bg-red-600 text-white rounded text-sm hover:bg-red-700"
        >
          重试
        </button>
      </div>
    );
  
  return (
    <ErrorBoundary FallbackComponent={errorFallback || DefaultErrorFallback}>
      <Suspense fallback={fallback || <div className="animate-pulse bg-gray-200 h-32 rounded"></div>}>
        <LazyComponent {...props} />
      </Suspense>
    </ErrorBoundary>
  );
};

// 防抖输入组件
export const DebouncedInput: React.FC<{
  value: string;
  onChange: (value: string) => void;
  delay?: number;
  placeholder?: string;
  className?: string;
}> = memo(({ value, onChange, delay = 300, placeholder, className }) => {
  const [localValue, setLocalValue] = React.useState(value);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  
  useEffect(() => {
    setLocalValue(value);
  }, [value]);
  
  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    setLocalValue(newValue);
    
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    
    timeoutRef.current = setTimeout(() => {
      onChange(newValue);
    }, delay);
  }, [onChange, delay]);
  
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);
  
  return (
    <input
      type="text"
      value={localValue}
      onChange={handleChange}
      placeholder={placeholder}
      className={className}
    />
  );
});

// 高阶组件：性能优化包装器
export const withPerformanceOptimization = <P extends object>(
  WrappedComponent: React.ComponentType<P>,
  options: {
    componentName: string;
    enableMemo?: boolean;
    enableMonitoring?: boolean;
    customShouldUpdate?: (prevProps: P, nextProps: P) => boolean;
  }
) => {
  const { componentName, enableMemo = true, enableMonitoring = false, customShouldUpdate } = options;
  
  const OptimizedComponent: React.FC<P> = (props) => {
    return (
      <PerformanceMonitor componentName={componentName} enableLogging={enableMonitoring}>
        <WrappedComponent {...props} />
      </PerformanceMonitor>
    );
  };
  
  if (enableMemo) {
    return memo(OptimizedComponent, customShouldUpdate);
  }
  
  return OptimizedComponent;
};

// 数据预加载组件
export const DataPreloader: React.FC<{
  urls: string[];
  onPreloadComplete?: (results: any[]) => void;
  children: React.ReactNode;
}> = ({ urls, onPreloadComplete, children }) => {
  const [isPreloading, setIsPreloading] = React.useState(true);
  
  useEffect(() => {
    const preloadData = async () => {
      try {
        const promises = urls.map(url => 
          fetch(url).then(res => res.json()).catch(err => ({ error: err.message }))
        );
        
        const results = await Promise.all(promises);
        onPreloadComplete?.(results);
      } catch (error) {
        console.error('Data preload failed:', error);
      } finally {
        setIsPreloading(false);
      }
    };
    
    if (urls.length > 0) {
      preloadData();
    } else {
      setIsPreloading(false);
    }
  }, [urls, onPreloadComplete]);
  
  if (isPreloading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <span className="ml-2 text-gray-600">预加载数据中...</span>
      </div>
    );
  }
  
  return <>{children}</>;
};

// 内存优化的表格组件
export const OptimizedTable: React.FC<{
  data: any[];
  columns: Array<{
    key: string;
    title: string;
    render?: (value: any, row: any, index: number) => React.ReactNode;
  }>;
  pageSize?: number;
  className?: string;
}> = memo(({ data, columns, pageSize = 50, className }) => {
  const [currentPage, setCurrentPage] = React.useState(0);
  
  const paginatedData = useMemo(() => {
    const start = currentPage * pageSize;
    return data.slice(start, start + pageSize);
  }, [data, currentPage, pageSize]);
  
  const totalPages = Math.ceil(data.length / pageSize);
  
  return (
    <div className={className}>
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              {columns.map((column) => (
                <th
                  key={column.key}
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                >
                  {column.title}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {paginatedData.map((row, index) => (
              <tr key={index} className="hover:bg-gray-50">
                {columns.map((column) => (
                  <td key={column.key} className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {column.render 
                      ? column.render(row[column.key], row, index)
                      : row[column.key]
                    }
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      {totalPages > 1 && (
        <div className="flex items-center justify-between px-4 py-3 bg-white border-t border-gray-200">
          <div className="flex items-center">
            <span className="text-sm text-gray-700">
              显示 {currentPage * pageSize + 1} 到 {Math.min((currentPage + 1) * pageSize, data.length)} 条，
              共 {data.length} 条记录
            </span>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setCurrentPage(Math.max(0, currentPage - 1))}
              disabled={currentPage === 0}
              className="px-3 py-1 text-sm bg-white border border-gray-300 rounded-md disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
            >
              上一页
            </button>
            <span className="text-sm text-gray-700">
              第 {currentPage + 1} 页，共 {totalPages} 页
            </span>
            <button
              onClick={() => setCurrentPage(Math.min(totalPages - 1, currentPage + 1))}
              disabled={currentPage === totalPages - 1}
              className="px-3 py-1 text-sm bg-white border border-gray-300 rounded-md disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
            >
              下一页
            </button>
          </div>
        </div>
      )}
    </div>
  );
});

// 性能统计显示组件
export const PerformanceStats: React.FC<{
  className?: string;
}> = ({ className }) => {
  const [stats, setStats] = React.useState<any>(null);
  
  useEffect(() => {
    const updateStats = () => {
      if ('memory' in performance) {
        const memory = (performance as any).memory;
        setStats({
          memoryUsage: (memory.usedJSHeapSize / (1024 * 1024)).toFixed(2),
          memoryLimit: (memory.jsHeapSizeLimit / (1024 * 1024)).toFixed(2),
          timestamp: new Date().toLocaleTimeString()
        });
      }
    };
    
    updateStats();
    const interval = setInterval(updateStats, 5000);
    
    return () => clearInterval(interval);
  }, []);
  
  if (!stats) return null;
  
  return (
    <div className={`text-xs text-gray-500 ${className}`}>
      内存使用: {stats.memoryUsage}MB / {stats.memoryLimit}MB
      <span className="ml-2">更新时间: {stats.timestamp}</span>
    </div>
  );
};