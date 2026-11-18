import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // 静态导出配置（可选，如果需要纯静态部署）
  // output: 'export',
  // distDir: 'dist',

  // 图片优化配置
  images: {
    unoptimized: true,
  },

  // 重写规则 - 代理API请求到后端
  async rewrites() {
    const apiUrl = process.env.API_URL || 'http://localhost:8000';
    return [
      {
        source: '/api/:path*',
        destination: `${apiUrl}/api/:path*`,
      },
    ];
  },

  // 环境变量
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
    NEXT_PUBLIC_WS_URL: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000',
  },
};

export default nextConfig;
