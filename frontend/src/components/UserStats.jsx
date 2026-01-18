import { Users, Shield, UserCheck } from "lucide-react";

export default function UserStats() {
  return (
    <div className="grid grid-cols-3 gap-6">
      <div className="bg-white border rounded-xl p-6 flex items-center gap-4">
        <div className="p-3 bg-blue-100 text-blue-600 rounded-full">
          <Users />
        </div>
        <div>
          <p className="text-sm text-slate-500">Total Users</p>
          <p className="text-2xl font-bold">3</p>
        </div>
      </div>

      <div className="bg-white border rounded-xl p-6 flex items-center gap-4">
        <div className="p-3 bg-red-100 text-red-600 rounded-full">
          <Shield />
        </div>
        <div>
          <p className="text-sm text-slate-500">Admins</p>
          <p className="text-2xl font-bold">2</p>
        </div>
      </div>

      <div className="bg-white border rounded-xl p-6 flex items-center gap-4">
        <div className="p-3 bg-green-100 text-green-600 rounded-full">
          <UserCheck />
        </div>
        <div>
          <p className="text-sm text-slate-500">Discipline Incharge</p>
          <p className="text-2xl font-bold">1</p>
        </div>
      </div>
    </div>
  );
}
