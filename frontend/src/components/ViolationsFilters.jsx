import { Search } from "lucide-react";

export default function ViolationsFilters() {
  return (
    <div className="bg-white dark:bg-gray-800 border dark:border-gray-700 rounded-xl p-4 flex flex-wrap gap-4 shadow-sm">
      <div className="relative flex-1 min-w-[250px]">
        <Search className="absolute left-3 top-2.5 text-slate-400 dark:text-gray-500" size={18} />
        <input
          placeholder="Search by ID, student, location..."
          className="w-full pl-10 pr-4 py-2 border dark:border-gray-600 rounded-lg text-sm dark:bg-gray-700 dark:text-white dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      {["All Types", "All Severity", "All Status", "All Cameras"].map(
        (label) => (
          <select
            key={label}
            className="px-4 py-2 border dark:border-gray-600 rounded-lg text-sm bg-white dark:bg-gray-700 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option>{label}</option>
          </select>
        )
      )}
    </div>
  );
}
