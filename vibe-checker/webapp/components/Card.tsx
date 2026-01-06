interface CardProps {
  children: React.ReactNode;
  className?: string;
  variant?: "default" | "error";
}

export default function Card({
  children,
  className = "",
  variant = "default",
}: CardProps) {
  const baseClasses = "rounded-lg border shadow-sm";

  const variantClasses = {
    default: "border-border-primary bg-bg-primary",
    error: "border-red-200 dark:border-red-900 bg-red-50 dark:bg-red-950",
  };

  return (
    <div className={`${baseClasses} ${variantClasses[variant]} ${className}`}>
      {children}
    </div>
  );
}
