import Topbar from "../../components/Topbar";
import UserStats from "../../components/UserStats";
import UsersTable from "../../components/UsersTable";

export default function Users() {
  return (
    <>
      <Topbar />

      <div className="p-6 space-y-6 bg-gray-50 dark:bg-gray-900 min-h-full">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Users Management</h1>
            <p className="text-slate-500 dark:text-gray-400">Manage system user accounts</p>
          </div>

          <button className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700">
            + Add User
          </button>
        </div>

        <UserStats />
        <UsersTable />
      </div>
    </>
  );
}
