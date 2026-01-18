import { Pencil } from "lucide-react";
import { users } from "../data/mockUsers";
import RoleBadge from "./RoleBadge";

export default function UsersTable() {
  return (
    <div className="bg-white border rounded-xl p-6">
      <h2 className="text-lg font-semibold mb-4">All Users</h2>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-slate-50 text-slate-600">
            <tr>
              <th className="text-left px-4 py-3">User ID</th>
              <th className="text-left px-4 py-3">Name</th>
              <th className="text-left px-4 py-3">Email</th>
              <th className="text-left px-4 py-3">Role</th>
              <th className="text-left px-4 py-3">Department</th>
              <th className="text-left px-4 py-3">Actions</th>
            </tr>
          </thead>

          <tbody>
            {users.map((u) => (
              <tr key={u.id} className="border-t">
                <td className="px-4 py-4 font-medium">{u.id}</td>
                <td className="px-4 py-4">{u.name}</td>
                <td className="px-4 py-4">{u.email}</td>
                <td className="px-4 py-4">
                  <RoleBadge role={u.role} />
                </td>
                <td className="px-4 py-4">{u.department}</td>
                <td className="px-4 py-4">
                  <button className="flex items-center gap-2 text-blue-600 hover:underline">
                    <Pencil size={16} /> Edit
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
