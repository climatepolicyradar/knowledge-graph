interface LoadingSpinnerProps {
  message?: string;
  className?: string;
}

export default function LoadingSpinner({
  message = "Loading...",
  className = "py-12",
}: LoadingSpinnerProps) {
  return (
    <div className={`text-center ${className}`}>
      <div className="inline-block h-6 w-6 animate-spin rounded-full border-2 border-neutral-300 border-t-neutral-900 dark:border-neutral-600 dark:border-t-neutral-100"></div>
      <p className="mt-4 text-sm text-neutral-600 dark:text-neutral-400">
        {message}
      </p>
    </div>
  );
}
