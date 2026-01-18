export default function StatCard({ title, value, icon, color }) {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl border dark:border-gray-700 p-6 flex justify-between items-center shadow-sm">
      <div>
        <p className="text-sm text-slate-500 dark:text-gray-400">{title}</p>
        <p className="text-3xl font-bold mt-1 text-gray-900 dark:text-white">{value}</p>
      </div>
      <div className={`p-3 rounded-full ${color}`}>
        {icon}
      </div>
    </div>
  );
}
