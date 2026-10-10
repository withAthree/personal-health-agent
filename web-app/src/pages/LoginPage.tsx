import { useMemo, useState } from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { HeartPulse, LockKeyhole, Phone } from 'lucide-react'
import { useForm } from 'react-hook-form'
import { z } from 'zod'

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
import { Field, FieldError, FieldGroup, FieldLabel } from '@/components/ui/field'
import { Input } from '@/components/ui/input'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useAuth } from '@/contexts/auth-context'
import { login, register } from '@/lib/api'

const appTitle = import.meta.env.VITE_APP_TITLE ?? 'Personal Health Agent'

/** 与后端 agent-app/src/agent_app/schemas/auth.py 的校验规则保持一致。 */
const PHONE_PATTERN = /^1[3-9]\d{9}$/
const PASSWORD_MIN_LENGTH = 8
const PASSWORD_MAX_LENGTH = 64

const phoneField = z
  .string()
  .trim()
  .regex(PHONE_PATTERN, '请输入正确的中国大陆手机号（11 位，1 开头）')

const registerPasswordField = z
  .string()
  .min(PASSWORD_MIN_LENGTH, `密码至少 ${String(PASSWORD_MIN_LENGTH)} 位`)
  .max(PASSWORD_MAX_LENGTH, `密码最多 ${String(PASSWORD_MAX_LENGTH)} 位`)

const baseSchema = z.object({
  phone: phoneField,
  password: z.string().min(1, '请输入密码'),
  confirmPassword: z.string(),
})

// 登录时密码只需要非空：长度规则由后端在注册时决定，登录不该因为长度不符就直接拦下
const loginSchema = baseSchema

const registerSchema = baseSchema
  .extend({ password: registerPasswordField })
  .refine((values) => values.password === values.confirmPassword, {
    message: '两次输入的密码不一致',
    path: ['confirmPassword'],
  })

type FormValues = z.infer<typeof baseSchema>

type Mode = 'login' | 'register'

const modes: readonly { key: Mode; label: string }[] = [
  { key: 'login', label: '登录' },
  { key: 'register', label: '注册' },
]

export function LoginPage() {
  const { signIn } = useAuth()
  const [mode, setMode] = useState<Mode>('login')
  const [serverError, setServerError] = useState<string | null>(null)

  const isRegister = mode === 'register'

  const resolver = useMemo(
    () => (isRegister ? zodResolver(registerSchema) : zodResolver(loginSchema)),
    [isRegister],
  )

  const form = useForm<FormValues>({
    resolver,
    defaultValues: { phone: '', password: '', confirmPassword: '' },
  })

  const { errors, isSubmitting } = form.formState

  function switchMode(value: string): void {
    if (value === mode) {
      return
    }
    setMode(value === 'register' ? 'register' : 'login')
    setServerError(null)
    form.clearErrors()
  }

  const submit = form.handleSubmit(async (values) => {
    setServerError(null)
    try {
      const credentials = { phone: values.phone.trim(), password: values.password }
      // 注册成功后接口直接返回令牌，等价于已登录
      const result = isRegister ? await register(credentials) : await login(credentials)
      // 保存登录态后路由门禁会放行，本组件随卸载；不做「回跳原页面」
      signIn(result)
    } catch (cause) {
      setServerError(cause instanceof Error ? cause.message : '操作失败，请稍后重试')
    }
  })

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
              noValidate
              onSubmit={(event) => {
                void submit(event)
              }}
            >
              <FieldGroup className="gap-4">
                <Field data-invalid={errors.phone !== undefined}>
                  <FieldLabel htmlFor="phone">手机号</FieldLabel>
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
                      aria-invalid={errors.phone !== undefined}
                      {...form.register('phone')}
                    />
                  </div>
                  <FieldError errors={[errors.phone]} />
                </Field>

                <Field data-invalid={errors.password !== undefined}>
                  <FieldLabel htmlFor="password">密码</FieldLabel>
                  <div className="relative">
                    <LockKeyhole
                      className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground"
                      aria-hidden="true"
                    />
                    <Input
                      id="password"
                      type="password"
                      autoComplete={isRegister ? 'new-password' : 'current-password'}
                      placeholder={
                        isRegister
                          ? `${String(PASSWORD_MIN_LENGTH)}–${String(PASSWORD_MAX_LENGTH)} 位`
                          : '请输入密码'
                      }
                      className="pl-8"
                      aria-invalid={errors.password !== undefined}
                      {...form.register('password')}
                    />
                  </div>
                  <FieldError errors={[errors.password]} />
                </Field>

                {isRegister ? (
                  <Field data-invalid={errors.confirmPassword !== undefined}>
                    <FieldLabel htmlFor="confirmPassword">确认密码</FieldLabel>
                    <div className="relative">
                      <LockKeyhole
                        className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground"
                        aria-hidden="true"
                      />
                      <Input
                        id="confirmPassword"
                        type="password"
                        autoComplete="new-password"
                        placeholder="再输入一次密码"
                        className="pl-8"
                        aria-invalid={errors.confirmPassword !== undefined}
                        {...form.register('confirmPassword')}
                      />
                    </div>
                    <FieldError errors={[errors.confirmPassword]} />
                  </Field>
                ) : null}

                {serverError !== null ? (
                  <Alert variant="destructive">
                    <AlertDescription>{serverError}</AlertDescription>
                  </Alert>
                ) : null}

                <Button type="submit" className="w-full" disabled={isSubmitting}>
                  {isSubmitting ? '处理中…' : isRegister ? '注册并登录' : '登录'}
                </Button>
              </FieldGroup>
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
