import { useState } from 'react'
import type { FormEvent } from 'react'
import { HeartPulse, LockKeyhole, Phone } from 'lucide-react'

import { Alert, AlertDescription } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useAuth } from '@/contexts/auth-context'
import { login, register } from '@/lib/api'

const appTitle = import.meta.env.VITE_APP_TITLE ?? 'Personal Health Agent'

/** 与后端 agent-app/src/agent_app/schemas/auth.py 的 PHONE_PATTERN 保持一致。 */
const PHONE_PATTERN = /^1[3-9]\d{9}$/
const PASSWORD_MIN_LENGTH = 8
const PASSWORD_MAX_LENGTH = 64

type Mode = 'login' | 'register'

const modes: readonly { key: Mode; label: string }[] = [
  { key: 'login', label: '登录' },
  { key: 'register', label: '注册' },
]

export function LoginPage() {
  const { signIn } = useAuth()
  const [mode, setMode] = useState<Mode>('login')
  const [phone, setPhone] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  const isRegister = mode === 'register'

  function switchMode(value: string): void {
    setMode(value === 'register' ? 'register' : 'login')
    setError(null)
  }

  function validate(): string | null {
    if (!PHONE_PATTERN.test(phone.trim())) {
      return '请输入正确的中国大陆手机号（11 位，1 开头）'
    }
    if (password.length < PASSWORD_MIN_LENGTH || password.length > PASSWORD_MAX_LENGTH) {
      return `密码长度需在 ${String(PASSWORD_MIN_LENGTH)}–${String(PASSWORD_MAX_LENGTH)} 位之间`
    }
    if (isRegister && password !== confirmPassword) {
      return '两次输入的密码不一致'
    }
    return null
  }

  async function submit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault()

    const message = validate()
    if (message !== null) {
      setError(message)
      return
    }

    setSubmitting(true)
    setError(null)
    try {
      const credentials = { phone: phone.trim(), password }
      // 注册成功后接口直接返回令牌，等价于已登录
      const result = isRegister ? await register(credentials) : await login(credentials)
      // 保存登录态后路由门禁会放行，本组件随卸载；不做「回跳原页面」
      signIn(result)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : '操作失败，请稍后重试')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="grid min-h-dvh place-items-center bg-muted/30 px-6 py-10 text-foreground">
      <div className="w-full max-w-sm space-y-6">
        <div className="flex flex-col items-center gap-2 text-center">
          <span className="grid size-10 place-items-center rounded-xl bg-primary text-primary-foreground">
            <HeartPulse className="size-5" aria-hidden="true" />
          </span>
          <h1 className="text-lg font-semibold tracking-tight">{appTitle}</h1>
          <p className="text-xs text-muted-foreground">基于多 Agent 架构的个人健康智能助手</p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>{isRegister ? '注册新账号' : '登录账号'}</CardTitle>
            <CardDescription>
              {isRegister ? '手机号即账号，注册成功即自动登录' : '输入手机号与密码继续'}
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-5">
            <Tabs value={mode} onValueChange={switchMode}>
              <TabsList className="grid w-full grid-cols-2">
                {modes.map((item) => (
                  <TabsTrigger key={item.key} value={item.key}>
                    {item.label}
                  </TabsTrigger>
                ))}
              </TabsList>
            </Tabs>

            <form
              className="space-y-4"
              onSubmit={(event) => {
                void submit(event)
              }}
            >
              <div className="space-y-1.5">
                <Label htmlFor="phone">手机号</Label>
                <div className="relative">
                  <Phone
                    className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground"
                    aria-hidden="true"
                  />
                  <Input
                    id="phone"
                    type="tel"
                    inputMode="numeric"
                    autoComplete="tel"
                    placeholder="11 位手机号"
                    className="pl-8"
                    value={phone}
                    onChange={(event) => {
                      setPhone(event.target.value)
                    }}
                  />
                </div>
              </div>

              <div className="space-y-1.5">
                <Label htmlFor="password">密码</Label>
                <div className="relative">
                  <LockKeyhole
                    className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground"
                    aria-hidden="true"
                  />
                  <Input
                    id="password"
                    type="password"
                    autoComplete={isRegister ? 'new-password' : 'current-password'}
                    placeholder={`${String(PASSWORD_MIN_LENGTH)}–${String(PASSWORD_MAX_LENGTH)} 位`}
                    className="pl-8"
                    value={password}
                    onChange={(event) => {
                      setPassword(event.target.value)
                    }}
                  />
                </div>
              </div>

              {isRegister ? (
                <div className="space-y-1.5">
                  <Label htmlFor="confirm">确认密码</Label>
                  <div className="relative">
                    <LockKeyhole
                      className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground"
                      aria-hidden="true"
                    />
                    <Input
                      id="confirm"
                      type="password"
                      autoComplete="new-password"
                      placeholder="再输入一次密码"
                      className="pl-8"
                      value={confirmPassword}
                      onChange={(event) => {
                        setConfirmPassword(event.target.value)
                      }}
                    />
                  </div>
                </div>
              ) : null}

              {error !== null ? (
                <Alert variant="destructive">
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              ) : null}

              <Button type="submit" className="w-full" disabled={submitting}>
                {submitting ? '处理中…' : isRegister ? '注册并登录' : '登录'}
              </Button>
            </form>
          </CardContent>

          <CardFooter className="justify-center">
            <p className="text-xs text-muted-foreground">登录状态保留 48 小时，期间免登录进入系统</p>
          </CardFooter>
        </Card>
      </div>
    </div>
  )
}
