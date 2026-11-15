import { useEffect } from 'react';
import { useRoute, useSettings } from './hooks';
import { setDigitMode } from './lib/format';
import { requestPersistence } from './db';
import { todayISO } from './lib/dates';
import { Today } from './pages/Today';
import { DayPage } from './pages/DayPage';
import { History } from './pages/History';
import { Star } from './pages/Star';
import { Experiences } from './pages/Experiences';
import { Balance } from './pages/Balance';
import { Settings } from './pages/Settings';

const NAV = [
  { path: 'today', label: 'امروز', icon: '☀' },
  { path: 'history', label: 'تاریخچه', icon: '▦' },
  { path: 'star', label: 'ستاره', icon: '✶' },
  { path: 'experiences', label: 'تجربه', icon: '◇' },
  { path: 'balance', label: 'تعادل', icon: '◎' },
  { path: 'settings', label: 'تنظیمات', icon: '⚙' },
];

export function App() {
  const route = useRoute();
  const settings = useSettings();
  setDigitMode(settings.digits); // before children render
  useEffect(() => { void requestPersistence(); }, []);

  const [page, arg, arg2] = route;
  let body;
  switch (page) {
    case 'day': body = <DayPage key={`${arg}-${arg2 ?? ''}`} date={arg ?? todayISO()} initialMode={arg2 === 'plan' ? 'plan' : undefined} />; break;
    case 'history': body = <History />; break;
    case 'star': body = <Star id={arg} />; break;
    case 'experiences': body = <Experiences />; break;
    case 'balance': body = <Balance />; break;
    case 'settings': body = <Settings />; break;
    default: body = <Today />;
  }
  const active = page === 'day' ? 'today' : page ?? 'today';

  return (
    <div className="app">
      <main>{body}</main>
      <nav className="tabbar">
        {NAV.map((n) => (
          <a key={n.path} href={`#/${n.path}`} className={active === n.path ? 'on' : ''}>
            <span className="ic" aria-hidden="true">{n.icon}</span>{n.label}
          </a>
        ))}
      </nav>
    </div>
  );
}
