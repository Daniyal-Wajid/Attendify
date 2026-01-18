import { Eye, User } from "lucide-react";

export default function StudentCard({ student }) {
  return (
    <div className="bg-white border rounded-xl p-5 flex flex-col">
      <div className="flex items-center gap-4">
        {student.avatar ? (
          <img
            src={student.avatar}
            alt={student.name}
            className="w-12 h-12 rounded-full object-cover"
          />
        ) : (
          <div className="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center">
            <User className="text-slate-400" />
          </div>
        )}

        <div>
          <div className="font-semibold">{student.name}</div>
          <div className="text-sm text-slate-500">{student.id}</div>
        </div>
      </div>

      <div className="mt-4 text-sm text-slate-600 space-y-1">
        <div>{student.department}</div>
        <div>{student.semester}</div>
      </div>

      <button className="mt-4 flex items-center justify-center gap-2 border rounded-lg py-2 text-sm font-medium hover:bg-slate-50">
        <Eye size={16} /> View Details
      </button>
    </div>
  );
}
