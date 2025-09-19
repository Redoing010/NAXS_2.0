import { useEffect, useRef, useCallback, useMemo } from 'react';
import { debounce, throttle } from 'lodash-es';

// 性能监控Hook
export const usePerformanceMonitor = () => {
  const metricsRef = useRef({
    renderCount: 0,
    lastRenderTime: 0,
    totalRenderTime: 0,
    memoryUsage: 0
  });

  const startRender = useCallback(() => {
    metricsRef.current.lastRenderTime = performance.now();
  }, []);

  const endRender = useCallback(() => {
    const renderTime = performance.now() - metricsRef.current.lastRenderTime;
    metricsRef.current.renderCount++;
    metricsRef.current.totalRenderTime += renderTime;
    
    // 记录内存使用情况（如果支持）
    if ('memory' in performance) {
      metricsRef.current.memoryUsage = (performance as any).memory.usedJSHeapSize;
    }
  }, []);

  const getMetrics = useCallback(() => {
    const { renderCount, totalRenderTime, memoryUsage } = metricsRef.current;
    return {
      renderCount,
      averageRenderTime: renderCount > 0 ? totalRenderTime / renderCount : 0,
      totalRenderTime,
      memoryUsage: memoryUsage / (1024 * 1024), // MB
      timestamp: Date.now()
    };
  }, []);

  const resetMetrics = useCallback(() => {
    metricsRef.current = {
      renderCount: 0,
      lastRenderTime: 0,
      totalRenderTime: 0,
      memoryUsage: 0
    };
  }, []);

  return {
    startRender,
    endRender,
    getMetrics,
    resetMetrics
  };
};

// 防抖Hook
export const useDebounce = <T extends (...args: any[]) => any>(
  callback: T,
  delay: number,
  deps: React.DependencyList = []
) => {
  return useMemo(
    () => debounce(callback, delay),
    [delay, ...deps]
  );
};

// 节流Hook
export const useThrottle = <T extends (...args: any[]) => any>(
  callback: T,
  delay: number,
  deps: React.DependencyList = []
) => {
  return useMemo(
    () => throttle(callback, delay),
    [delay, ...deps]
  );
};

// 虚拟滚动Hook
export const useVirtualScroll = ({
  itemCount,
  itemHeight,
  containerHeight,
  overscan = 5
}: {
  itemCount: number;
  itemHeight: number;
  containerHeight: number;
  overscan?: number;
}) => {
  const scrollTopRef = useRef(0);
  
  const visibleRange = useMemo(() => {
    const startIndex = Math.floor(scrollTopRef.current / itemHeight);
    const endIndex = Math.min(
      itemCount - 1,
      Math.ceil((scrollTopRef.current + containerHeight) / itemHeight)
    );
    
    return {
      start: Math.max(0, startIndex - overscan),
      end: Math.min(itemCount - 1, endIndex + overscan)
    };
  }, [itemCount, itemHeight, containerHeight, overscan]);

  const handleScroll = useCallback((scrollTop: number) => {
    scrollTopRef.current = scrollTop;
  }, []);

  const totalHeight = itemCount * itemHeight;
  const offsetY = visibleRange.start * itemHeight;

  return {
    visibleRange,
    totalHeight,
    offsetY,
    handleScroll
  };
};

// 图片懒加载Hook
export const useLazyImage = (src: string, placeholder?: string) => {
  const imgRef = useRef<HTMLImageElement>(null);
  const observerRef = useRef<IntersectionObserver | null>(null);
  
  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;

    observerRef.current = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          img.src = src;
          observerRef.current?.disconnect();
        }
      },
      { threshold: 0.1 }
    );

    observerRef.current.observe(img);

    return () => {
      observerRef.current?.disconnect();
    };
  }, [src]);

  return {
    ref: imgRef,
    src: placeholder || 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMSIgaGVpZ2h0PSIxIiB2aWV3Qm94PSIwIDAgMSAxIiBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxyZWN0IHdpZHRoPSIxIiBoZWlnaHQ9IjEiIGZpbGw9IiNGNUY1RjUiLz48L3N2Zz4='
  };
};

// 内存泄漏检测Hook
export const useMemoryLeak = (componentName: string) => {
  const mountTimeRef = useRef<number>(Date.now());
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  
  useEffect(() => {
    // 组件挂载时记录
    console.log(`[Memory] ${componentName} mounted at ${new Date().toISOString()}`);
    
    // 定期检查内存使用情况
    if ('memory' in performance) {
      timerRef.current = setInterval(() => {
        const memory = (performance as any).memory;
        const memoryUsage = memory.usedJSHeapSize / (1024 * 1024);
        
        if (memoryUsage > 100) { // 超过100MB时警告
          console.warn(`[Memory] High memory usage in ${componentName}: ${memoryUsage.toFixed(2)}MB`);
        }
      }, 10000); // 每10秒检查一次
    }
    
    return () => {
      // 组件卸载时清理
      const lifeTime = Date.now() - mountTimeRef.current;
      console.log(`[Memory] ${componentName} unmounted after ${lifeTime}ms`);
      
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, [componentName]);
};

// 组件渲染优化Hook
export const useRenderOptimization = () => {
  const renderCountRef = useRef(0);
  const lastPropsRef = useRef<any>(null);
  
  const trackRender = useCallback((props: any, componentName: string) => {
    renderCountRef.current++;
    
    if (process.env.NODE_ENV === 'development') {
      // 检查不必要的重渲染
      if (lastPropsRef.current && JSON.stringify(lastPropsRef.current) === JSON.stringify(props)) {
        console.warn(`[Render] Unnecessary re-render in ${componentName} (render #${renderCountRef.current})`);
      }
      
      lastPropsRef.current = props;
    }
  }, []);
  
  const getRenderCount = useCallback(() => renderCountRef.current, []);
  
  return {
    trackRender,
    getRenderCount
  };
};

// 网络请求优化Hook
export const useOptimizedFetch = () => {
  const abortControllerRef = useRef<AbortController | null>(null);
  const cacheRef = useRef<Map<string, { data: any; timestamp: number }>>(new Map());
  
  const fetchWithCache = useCallback(async (
    url: string, 
    options: RequestInit = {},
    cacheTime: number = 5 * 60 * 1000 // 5分钟缓存
  ) => {
    // 检查缓存
    const cached = cacheRef.current.get(url);
    if (cached && Date.now() - cached.timestamp < cacheTime) {
      return cached.data;
    }
    
    // 取消之前的请求
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    
    // 创建新的AbortController
    abortControllerRef.current = new AbortController();
    
    try {
      const response = await fetch(url, {
        ...options,
        signal: abortControllerRef.current.signal
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      // 缓存结果
      cacheRef.current.set(url, {
        data,
        timestamp: Date.now()
      });
      
      return data;
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        console.log('Request aborted');
        return null;
      }
      throw error;
    }
  }, []);
  
  const clearCache = useCallback(() => {
    cacheRef.current.clear();
  }, []);
  
  useEffect(() => {
    return () => {
      // 组件卸载时取消请求
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);
  
  return {
    fetchWithCache,
    clearCache
  };
};

// Web Worker Hook
export const useWebWorker = (workerScript: string) => {
  const workerRef = useRef<Worker | null>(null);
  const callbacksRef = useRef<Map<string, (data: any) => void>>(new Map());
  
  useEffect(() => {
    // 创建Web Worker
    workerRef.current = new Worker(workerScript);
    
    // 监听消息
    workerRef.current.onmessage = (event) => {
      const { id, data, error } = event.data;
      const callback = callbacksRef.current.get(id);
      
      if (callback) {
        if (error) {
          console.error('Worker error:', error);
        } else {
          callback(data);
        }
        callbacksRef.current.delete(id);
      }
    };
    
    return () => {
      // 清理Worker
      if (workerRef.current) {
        workerRef.current.terminate();
      }
    };
  }, [workerScript]);
  
  const postMessage = useCallback((data: any, callback?: (result: any) => void) => {
    if (!workerRef.current) return;
    
    const id = Math.random().toString(36).substr(2, 9);
    
    if (callback) {
      callbacksRef.current.set(id, callback);
    }
    
    workerRef.current.postMessage({ id, data });
  }, []);
  
  return {
    postMessage,
    isReady: !!workerRef.current
  };
};