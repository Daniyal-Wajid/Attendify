import { Eye, User } from "lucide-react";

export default function StudentCard({ student }) {
  return (
    <div className="bg-white dark:bg-gray-800 border dark:border-gray-700 rounded-xl p-5 flex flex-col shadow-sm">
      <div className="flex items-center gap-4">
        {student.avatar ? (
          <img
            src={student.avatar}
            alt={student.name}
            className="w-12 h-12 rounded-full object-cover"
          />
        ) : (
          <div className="w-12 h-12 rounded-full bg-slate-100 dark:bg-gray-700 flex items-center justify-center">
            <User className="text-slate-400 dark:text-gray-400" />
          </div>
        )}

        <div>
          <div className="font-semibold text-gray-900 dark:text-white">{student.name}</div>
          <div className="text-sm text-slate-500 dark:text-gray-400">{student.id}</div>
        </div>
      </div>

      <div className="mt-4 text-sm text-slate-600 dark:text-gray-300 space-y-1">
        <div>{student.department}</div>
        <div>{student.semester}</div>
      </div>

      <button className="mt-4 flex items-center justify-center gap-2 border dark:border-gray-600 rounded-lg py-2 text-sm font-medium hover:bg-slate-50 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 transition-colors">
        <Eye size={16} /> View Details
      </button>
    </div>
  );
}
