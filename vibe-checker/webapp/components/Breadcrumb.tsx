import Link from "next/link";
import MaterialIcon from "@/components/MaterialIcon";

interface BreadcrumbProps {
  href: string;
  children: React.ReactNode;
}

export default function Breadcrumb({ href, children }: BreadcrumbProps) {
  return (
    <div className="mb-6">
      <Link
        href={href}
        className="inline-flex items-center gap-2 text-sm text-secondary transition-colors hover:text-primary"
      >
        <MaterialIcon name="arrow_back" size={16} />
        {children}
      </Link>
    </div>
  );
}