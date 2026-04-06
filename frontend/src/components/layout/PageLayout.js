import React from "react";
import { Header } from "./Header";
import { Sidebar } from "./Sidebar";

export function PageLayout({ children }) {
  return (
    <div className="app-shell">
      <Header />
      <div className="app-body">
        <Sidebar />
        <main className="app-main">{children}</main>
      </div>
    </div>
  );
}
