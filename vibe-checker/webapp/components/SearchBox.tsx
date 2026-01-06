import { useState, useEffect, useRef } from "react";
import { Input } from "@base-ui-components/react/input";
import MaterialIcon from "@/components/MaterialIcon";

interface SearchBoxProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  className?: string;
  debounceMs?: number;
}

export default function SearchBox({
  value,
  onChange,
  placeholder = "Search...",
  className = "",
  debounceMs = 300,
}: SearchBoxProps) {
  const [localValue, setLocalValue] = useState(value);
  const debounceTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Update local value when external value changes
  useEffect(() => {
    setLocalValue(value);
  }, [value]);

  // Debounced change handler
  const handleChange = (newValue: string) => {
    setLocalValue(newValue);

    // Clear existing timeout
    if (debounceTimeoutRef.current) {
      clearTimeout(debounceTimeoutRef.current);
    }

    // Set new timeout for onChange
    debounceTimeoutRef.current = setTimeout(() => {
      onChange(newValue);
    }, debounceMs);
  };

  const handleClear = () => {
    setLocalValue("");
    onChange("");
  };

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (debounceTimeoutRef.current) {
        clearTimeout(debounceTimeoutRef.current);
      }
    };
  }, []);

  return (
    <div className={`relative ${className}`}>
      <Input
        value={localValue}
        onChange={(e) => handleChange(e.target.value)}
        className={`input w-full !pl-9 py-2 text-sm ${localValue ? '!pr-9' : '!pr-4'}`}
        placeholder={placeholder}
      />
      <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
        <MaterialIcon name="search" size={16} className="text-secondary" />
      </div>
      {localValue && (
        <button
          onClick={handleClear}
          className="absolute inset-y-0 right-0 pr-3 flex items-center text-secondary hover:text-primary transition-colors"
        >
          <MaterialIcon name="close" size={16} />
        </button>
      )}
    </div>
  );
}