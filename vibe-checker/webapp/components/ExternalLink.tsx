import MaterialIcon from "@/components/MaterialIcon";

interface ExternalLinkProps {
  href: string;
  children: React.ReactNode;
  className?: string;
}

export default function ExternalLink({ href, children, className = "" }: ExternalLinkProps) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className={`inline-flex items-center gap-1 transition-colors ${className}`}
    >
      {children}
      <MaterialIcon name="open_in_new" size={16} />
    </a>
  );
}