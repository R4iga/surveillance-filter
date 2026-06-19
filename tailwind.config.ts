import type { Config } from 'tailwindcss';
const config: Config = {
  content: ['./app/**/*.{js,ts,jsx,tsx,mdx}', './src/**/*.{js,ts,jsx,tsx,mdx}'],
  theme: {
    extend: {
      colors: {
        bg: '#05080c',
        panel: '#0a0f16',
        fg: '#cfe8d4',
        muted: '#5a6b5f',
        border: '#13201a',
        accent: '#39ff7a',
        'accent-dim': '#1a8f43',
        warn: '#ffcc33',
        danger: '#ff3b3b',
      },
      fontFamily: {
        mono: ['var(--font-mono)', 'ui-monospace', 'monospace'],
        display: ['var(--font-display)', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
};
export default config;
