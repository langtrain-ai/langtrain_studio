export const transitions = {
    spring: {
        type: "spring",
        stiffness: 260,
        damping: 20
    },
    gentle: {
        type: "spring",
        stiffness: 100,
        damping: 15
    },
    default: {
        duration: 0.3,
        ease: [0.4, 0, 0.2, 1]
    }
} as const;

export const variants = {
    fadeIn: {
        initial: { opacity: 0 },
        animate: { opacity: 1 },
        exit: { opacity: 0 }
    },
    slideUp: {
        initial: { opacity: 0, y: 15 },
        animate: { opacity: 1, y: 0 },
        exit: { opacity: 0, y: 15 }
    },
    scaleIn: {
        initial: { opacity: 0, scale: 0.98 },
        animate: { opacity: 1, scale: 1 },
        exit: { opacity: 0, scale: 0.98 }
    }
} as const;
