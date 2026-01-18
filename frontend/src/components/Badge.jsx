export default function Badge({ text, variant }) {
  const styles = {
    MED: "bg-blue-100 text-blue-600",
    HIGH: "bg-red-100 text-red-600",
    Unverified: "bg-orange-100 text-orange-600",
    Rejected: "bg-red-100 text-red-600",
  };

  return (
    <span
      className={`px-3 py-1 rounded-full text-xs font-semibold ${
        styles[variant]
      }`}
    >
      {text}
    </span>
  );
}
