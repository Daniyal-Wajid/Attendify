export default function RoleBadge({ role }) {
  const styles = {
    Admin: "bg-red-100 text-red-600",
    "Discipline Incharge": "bg-blue-100 text-blue-600",
  };

  return (
    <span className={`px-3 py-1 rounded-full text-xs font-semibold ${styles[role]}`}>
      {role}
    </span>
  );
}
