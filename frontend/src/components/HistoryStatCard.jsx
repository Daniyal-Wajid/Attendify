export default function HistoryStatCard({ icon, label, value }) {
  return (
    <div className="bg-white dark:bg-gray-800 border dark:border-gray-700 rounded-xl p-6 flex items-center gap-4 shadow-sm">
      <div className="text-2xl">{icon}</div>
      <div>
        <p className="text-slate-500 dark:text-gray-400 text-sm">{label}</p>
        <p className="text-2xl font-bold text-gray-900 dark:text-white">{value}</p>
      </div>
    </div>
  );
}
