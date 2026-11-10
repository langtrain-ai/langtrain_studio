export const colors = {
    // Brand Palette
    brand: {
        primary: "#1A73E8",    // Professional Blue
        secondary: "#7C3AED",  // Vibrant Purple
        electric: "#00F2FF",   // Highlight Cyan
        vibrant: "#FF00E5",    // Pink Accent
    },

    // UI Foundation (Desktop Optimized)
    base: {
        black: "#000000",
        white: "#FFFFFF",
        bg: "#0D0D0F",         // From Swift Design
        sidebar: "#0F0F11",    // Sidebar specific
        bgCard: "#17171A",
        bgElevated: "#1F1F22",
        border: "#27272A",
    },

    // Status
    status: {
        success: "#22C55E",
        warning: "#F97316",
        error: "#EF4444",
        info: "#3B82F6",
    },

    // Glassmorphism
    glass: {
        background: "rgba(23, 23, 26, 0.7)",
        border: "1px solid rgba(255, 255, 255, 0.08)",
        blur: "backdrop-blur-xl",
    }
} as const;
