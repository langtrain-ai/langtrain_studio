import { colors } from "./colors";

export const cards = {
    base: {
        background: colors.base.bgCard,
        border: `1px solid ${colors.base.border}`,
        radius: "12px",
    },

    hover: {
        background: colors.base.bgElevated,
        border: "1px solid rgba(255, 255, 255, 0.15)",
    },

    glass: {
        background: "rgba(23, 23, 26, 0.4)",
        blur: "backdrop-blur-xl",
        border: "1px solid rgba(255, 255, 255, 0.08)",
    }
} as const;
