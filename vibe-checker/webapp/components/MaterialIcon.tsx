interface MaterialIconProps {
  name: string;
  className?: string;
  size?: number;
}

export default function MaterialIcon({
  name,
  className = "",
  size = 24
}: MaterialIconProps) {
  return (
    <span
      className={`material-symbols-rounded select-none ${className}`}
      style={{ fontSize: `${size}px` }}
    >
      {name}
    </span>
  );
}