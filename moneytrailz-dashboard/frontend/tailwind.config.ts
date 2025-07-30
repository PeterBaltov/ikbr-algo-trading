import type { Config } from "tailwindcss"

const config: Config = {
  darkMode: "class",
  content: [
    './src/pages/**/*.{ts,tsx}',
    './src/components/**/*.{ts,tsx}',
    './src/app/**/*.{ts,tsx}',
    './src/**/*.{ts,tsx}',
  ],
  prefix: "",
  theme: {
    container: {
      center: true,
      padding: "2rem",
      screens: {
        "2xl": "1400px",
      },
    },
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        // Professional Trading Colors (IB-inspired)
        profit: {
          DEFAULT: "hsl(142, 76%, 45%)", // IB Green
          foreground: "hsl(0, 0%, 100%)",
          light: "hsl(142, 76%, 85%)",
          dark: "hsl(142, 76%, 35%)",
        },
        loss: {
          DEFAULT: "hsl(0, 84%, 60%)", // IB Red  
          foreground: "hsl(0, 0%, 100%)",
          light: "hsl(0, 84%, 85%)",
          dark: "hsl(0, 84%, 50%)",
        },
        neutral: {
          DEFAULT: "hsl(45, 100%, 51%)", // IB Yellow/Gold
          foreground: "hsl(0, 0%, 0%)",
          light: "hsl(45, 100%, 85%)",
          dark: "hsl(45, 100%, 41%)",
        },
        // IB-specific trading colors
        bid: {
          DEFAULT: "hsl(45, 100%, 51%)", // Yellow for bid prices
          foreground: "hsl(0, 0%, 0%)",
        },
        ask: {
          DEFAULT: "hsl(142, 76%, 45%)", // Green for ask prices
          foreground: "hsl(0, 0%, 100%)",
        },
        buy: {
          DEFAULT: "hsl(217, 91%, 60%)", // IB Blue for buy
          foreground: "hsl(0, 0%, 100%)",
        },
        sell: {
          DEFAULT: "hsl(0, 84%, 60%)", // Red for sell
          foreground: "hsl(0, 0%, 100%)",
        },
        trading: {
          positive: "hsl(142, 76%, 45%)",
          negative: "hsl(0, 84%, 60%)",
          neutral: "hsl(45, 100%, 51%)",
          unchanged: "hsl(210, 20%, 60%)",
          last: "hsl(199, 89%, 48%)", // Light blue for last price
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      keyframes: {
        "accordion-down": {
          from: { height: "0" },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: "0" },
        },
        "pulse-profit": {
          "0%, 100%": { 
            backgroundColor: "hsl(142, 76%, 36%)",
            transform: "scale(1)"
          },
          "50%": { 
            backgroundColor: "hsl(142, 76%, 45%)",
            transform: "scale(1.02)"
          },
        },
        "pulse-loss": {
          "0%, 100%": { 
            backgroundColor: "hsl(0, 84%, 60%)",
            transform: "scale(1)"
          },
          "50%": { 
            backgroundColor: "hsl(0, 84%, 70%)",
            transform: "scale(1.02)"
          },
        },
        "fade-in": {
          "0%": { opacity: "0", transform: "translateY(10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "slide-in": {
          "0%": { transform: "translateX(-100%)" },
          "100%": { transform: "translateX(0)" },
        },
        "bounce-gentle": {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-2px)" },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
        "pulse-profit": "pulse-profit 2s ease-in-out infinite",
        "pulse-loss": "pulse-loss 2s ease-in-out infinite",
        "fade-in": "fade-in 0.5s ease-out",
        "slide-in": "slide-in 0.3s ease-out",
        "bounce-gentle": "bounce-gentle 1s ease-in-out infinite",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
} satisfies Config

export default config 
