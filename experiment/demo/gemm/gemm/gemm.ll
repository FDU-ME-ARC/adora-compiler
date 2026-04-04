; ModuleID = 'gemm.c'
source_filename = "gemm.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @gemm(ptr noundef %C, ptr noundef %A, ptr noundef %B) #0 {
entry:
  %C.addr = alloca ptr, align 8
  %A.addr = alloca ptr, align 8
  %B.addr = alloca ptr, align 8
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %k = alloca i32, align 4
  %alpha = alloca double, align 8
  %beta = alloca double, align 8
  store ptr %C, ptr %C.addr, align 8
  store ptr %A, ptr %A.addr, align 8
  store ptr %B, ptr %B.addr, align 8
  store double 1.500000e+00, ptr %alpha, align 8
  store double 1.200000e+00, ptr %beta, align 8
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc32, %entry
  %0 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %0, 1000
  br i1 %cmp, label %for.body, label %for.end34

for.body:                                         ; preds = %for.cond
  store i32 0, ptr %j, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %1 = load i32, ptr %j, align 4
  %cmp2 = icmp slt i32 %1, 1100
  br i1 %cmp2, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %2 = load double, ptr %beta, align 8
  %3 = load ptr, ptr %C.addr, align 8
  %4 = load i32, ptr %i, align 4
  %idxprom = sext i32 %4 to i64
  %arrayidx = getelementptr inbounds [1100 x double], ptr %3, i64 %idxprom
  %5 = load i32, ptr %j, align 4
  %idxprom4 = sext i32 %5 to i64
  %arrayidx5 = getelementptr inbounds [1100 x double], ptr %arrayidx, i64 0, i64 %idxprom4
  %6 = load double, ptr %arrayidx5, align 8
  %mul = fmul double %6, %2
  store double %mul, ptr %arrayidx5, align 8
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %7 = load i32, ptr %j, align 4
  %inc = add nsw i32 %7, 1
  store i32 %inc, ptr %j, align 4
  br label %for.cond1, !llvm.loop !6

for.end:                                          ; preds = %for.cond1
  store i32 0, ptr %k, align 4
  br label %for.cond6

for.cond6:                                        ; preds = %for.inc29, %for.end
  %8 = load i32, ptr %k, align 4
  %cmp7 = icmp slt i32 %8, 1200
  br i1 %cmp7, label %for.body8, label %for.end31

for.body8:                                        ; preds = %for.cond6
  store i32 0, ptr %j, align 4
  br label %for.cond9

for.cond9:                                        ; preds = %for.inc26, %for.body8
  %9 = load i32, ptr %j, align 4
  %cmp10 = icmp slt i32 %9, 1100
  br i1 %cmp10, label %for.body11, label %for.end28

for.body11:                                       ; preds = %for.cond9
  %10 = load double, ptr %alpha, align 8
  %11 = load ptr, ptr %A.addr, align 8
  %12 = load i32, ptr %i, align 4
  %idxprom12 = sext i32 %12 to i64
  %arrayidx13 = getelementptr inbounds [1200 x double], ptr %11, i64 %idxprom12
  %13 = load i32, ptr %k, align 4
  %idxprom14 = sext i32 %13 to i64
  %arrayidx15 = getelementptr inbounds [1200 x double], ptr %arrayidx13, i64 0, i64 %idxprom14
  %14 = load double, ptr %arrayidx15, align 8
  %mul16 = fmul double %10, %14
  %15 = load ptr, ptr %B.addr, align 8
  %16 = load i32, ptr %k, align 4
  %idxprom17 = sext i32 %16 to i64
  %arrayidx18 = getelementptr inbounds [1100 x double], ptr %15, i64 %idxprom17
  %17 = load i32, ptr %j, align 4
  %idxprom19 = sext i32 %17 to i64
  %arrayidx20 = getelementptr inbounds [1100 x double], ptr %arrayidx18, i64 0, i64 %idxprom19
  %18 = load double, ptr %arrayidx20, align 8
  %19 = load ptr, ptr %C.addr, align 8
  %20 = load i32, ptr %i, align 4
  %idxprom22 = sext i32 %20 to i64
  %arrayidx23 = getelementptr inbounds [1100 x double], ptr %19, i64 %idxprom22
  %21 = load i32, ptr %j, align 4
  %idxprom24 = sext i32 %21 to i64
  %arrayidx25 = getelementptr inbounds [1100 x double], ptr %arrayidx23, i64 0, i64 %idxprom24
  %22 = load double, ptr %arrayidx25, align 8
  %23 = call double @llvm.fmuladd.f64(double %mul16, double %18, double %22)
  store double %23, ptr %arrayidx25, align 8
  br label %for.inc26

for.inc26:                                        ; preds = %for.body11
  %24 = load i32, ptr %j, align 4
  %inc27 = add nsw i32 %24, 1
  store i32 %inc27, ptr %j, align 4
  br label %for.cond9, !llvm.loop !8

for.end28:                                        ; preds = %for.cond9
  br label %for.inc29

for.inc29:                                        ; preds = %for.end28
  %25 = load i32, ptr %k, align 4
  %inc30 = add nsw i32 %25, 1
  store i32 %inc30, ptr %k, align 4
  br label %for.cond6, !llvm.loop !9

for.end31:                                        ; preds = %for.cond6
  br label %for.inc32

for.inc32:                                        ; preds = %for.end31
  %26 = load i32, ptr %i, align 4
  %inc33 = add nsw i32 %26, 1
  store i32 %inc33, ptr %i, align 4
  br label %for.cond, !llvm.loop !10

for.end34:                                        ; preds = %for.cond
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #1

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"clang version 18.0.0"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = distinct !{!8, !7}
!9 = distinct !{!9, !7}
!10 = distinct !{!10, !7}
