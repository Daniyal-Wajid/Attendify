import Topbar from "../../components/Topbar";
import NotificationStats from "../../components/NotificationStats";
import NotificationsList from "../../components/NotificationsList";

export default function Notifications() {
  return (
    <>
      <Topbar />

      <div className="p-6 space-y-6 bg-gray-50 dark:bg-gray-900 min-h-full">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Notifications</h1>
            <p className="text-slate-500 dark:text-gray-400">6 unread notifications</p>
          </div>

          <button className="px-4 py-2 border rounded-lg text-sm hover:bg-slate-100">
            ✓ Mark All Read
          </button>
        </div>

        <NotificationStats />
        <NotificationsList />
      </div>
    </>
  );
}
